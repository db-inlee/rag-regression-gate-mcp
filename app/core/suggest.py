"""Rule-based, deterministic remediation suggestions (M2). NO LLM.

Pipeline (phase-mcp §M2):
  1. regressed failure-mode metric → pipeline stage (catalog 표1).
  2. config diff reverse-trace (candidate vs baseline run.jsonl header) → if a changed
     param is a known cause of that mode (catalog 표2), propose REVERT as priority 1.
  3. slice concentration (candidate attribution.jsonl) → emphasize slice-specific
     techniques (catalog 표3).
  4. offer ONLY techniques whose stage matches the mode (no CoT for retrieval_miss).

Every suggestion carries the suggestion-only footer: never auto-applied; re-verify
with this gate after a human applies a change. The catalog
(docs/remediation_catalog.md) is the single source of truth — parsed, not hardcoded.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

CATALOG_PATH = Path("docs/remediation_catalog.md")
FOOTER = "(자동 적용 안 함 — 사람이 적용 후 이 게이트로 재검증.)"

# detect metric name → failure mode. answerable/no_answer accuracy are symptoms that
# route to the failure mode behind them (retrieval/grounding ↔ refusal).
_METRIC_TO_MODE = {
    "retrieval_miss": "retrieval_miss",
    "retrieval_success_strict": "retrieval_miss",
    "retrieval_success_value_present": "retrieval_miss",
    "hallucination": "hallucination",
    "over_answer": "over_answer",
    "over_answer_rate": "over_answer",
    "no_answer_accuracy": "over_answer",
}
# config params worth diffing (ignore derived: signature/collection/persist/etc.)
_SEMANTIC_PARAMS = ("top_k", "chunk_size", "chunk_overlap", "embedding_model",
                    "generation_model", "reranker_enabled")
_STAGE_KR = {"retrieval": "검색", "grounding": "그라운딩", "refusal": "거부",
             "generation_reasoning": "생성·추론"}


# --- catalog parsing (single source of truth) ------------------------------- #

def _parse_catalog(path: Path = CATALOG_PATH) -> dict:
    """Parse the three pipe-tables. Returns {techniques, causes, slice_emphasis}."""
    techniques: dict[str, dict] = {}      # mode -> {stage, techs:[...]}
    causes: list[tuple] = []              # (param, bad_direction, mode)
    slice_emphasis: dict[str, str] = {}   # slice -> emphasis
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):   # separator row
            continue
        head = cells[0].lower()
        if head == "failure_mode":
            section = "tech"; continue
        if head == "param":
            section = "cause"; continue
        if head == "slice":
            section = "slice"; continue
        if section == "tech" and len(cells) == 3:
            techniques[cells[0]] = {"stage": cells[1],
                                    "techs": [t.strip() for t in cells[2].split(";") if t.strip()]}
        elif section == "cause" and len(cells) == 3:
            causes.append((cells[0], cells[1], cells[2]))
        elif section == "slice" and len(cells) == 2:
            slice_emphasis[cells[0]] = cells[1]
    return {"techniques": techniques, "causes": causes, "slice_emphasis": slice_emphasis}


# --- inputs from run dirs --------------------------------------------------- #

def _load_config(run_dir: Path) -> dict:
    """Full config from the run.jsonl header (fallback: meta.json['config'])."""
    header = json.loads((run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()[0])
    if header.get("config"):
        return header["config"]
    meta = run_dir / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8")).get("config", {}) if meta.exists() else {}


def _direction(b, c) -> str:
    if isinstance(b, bool) or isinstance(c, bool):
        return "disable" if (b and not c) else ("enable" if (c and not b) else "same")
    if isinstance(b, (int, float)) and isinstance(c, (int, float)):
        return "decrease" if c < b else ("increase" if c > b else "same")
    return "change" if b != c else "same"


def _config_diff(base: dict, cand: dict) -> dict:
    """{param: (base_val, cand_val, direction)} for semantic params that changed."""
    out = {}
    for p in _SEMANTIC_PARAMS:
        if p in base and p in cand and base[p] != cand[p]:
            out[p] = (base[p], cand[p], _direction(base[p], cand[p]))
    return out


def _slice_concentration(cand_dir: Path, mode: str, frac: float = 0.6) -> str | None:
    """Dominant slice among the candidate's cases attributed to `mode` (≥frac)."""
    attr_path = cand_dir / "attribution.jsonl"
    if not attr_path.exists():
        return None
    slices = [a.get("slice") for a in
              (json.loads(l) for l in attr_path.read_text(encoding="utf-8").splitlines() if l.strip())
              if a.get("primary_failure") == mode]
    if not slices:
        return None
    top, n = Counter(slices).most_common(1)[0]
    return top if n / len(slices) >= frac else None


# --- main ------------------------------------------------------------------- #

def build_suggestions(report: dict, baseline_dir: str | Path, candidate_dir: str | Path) -> list[str]:
    """Deterministic suggestions for the regressed failure modes (suggestion-only)."""
    cat = _parse_catalog()
    base_cfg = _load_config(Path(baseline_dir))
    cand_cfg = _load_config(Path(candidate_dir))
    diff = _config_diff(base_cfg, cand_cfg)

    # regressed failure-mode metrics (FAIL + WARN), mapped to modes, keep worst row per mode
    regressed = [r for r in report["results"] if r["direction"] in ("regression", "warn")]
    rows_by_mode: dict[str, dict] = {}
    for r in regressed:
        mode = _METRIC_TO_MODE.get(r["metric"])
        if mode and (mode not in rows_by_mode or abs(r["delta"]) > abs(rows_by_mode[mode]["delta"])):
            rows_by_mode[mode] = r

    suggestions: list[str] = []
    # modes with a config-revert candidate first (most actionable)
    def has_revert(mode):
        return any(p in diff and diff[p][2] == bad for (p, bad, m) in cat["causes"] if m == mode)
    for mode in sorted(rows_by_mode, key=lambda m: (not has_revert(m), m)):
        r = rows_by_mode[mode]
        entry = cat["techniques"].get(mode)
        if not entry:
            continue
        stage_kr = _STAGE_KR.get(entry["stage"], entry["stage"])
        head = (f"[{mode}] {stage_kr} 단계 회귀: {r['metric']} "
                f"{_fmt(r['baseline'])}→{_fmt(r['candidate'])} (Δ{r['delta']:+.4f}).")

        # step 2: config-diff revert (priority 1)
        revert = ""
        for (p, bad, m) in cat["causes"]:
            if m == mode and p in diff and diff[p][2] == bad:
                b, c, _ = diff[p]
                revert = f" 원인 후보: {p} {b}→{c}. → 우선 되돌림({p} {c}→{b}) 검토."
                break
        if not revert:
            revert = " (candidate config에서 직접적 원인 파라미터는 안 보임.)"

        # step 4: stage-matched techniques only (no cross-stage leakage)
        techs = entry["techs"][:3]
        tail = f" 그래도 부족하면: {', '.join(techs)}."

        # step 3: slice concentration emphasis (optional)
        conc = _slice_concentration(Path(candidate_dir), mode)
        emph = f" [슬라이스 집중] {cat['slice_emphasis'][conc]}." if conc and conc in cat["slice_emphasis"] else ""

        suggestions.append(head + revert + tail + emph + " " + FOOTER)
    return suggestions


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
