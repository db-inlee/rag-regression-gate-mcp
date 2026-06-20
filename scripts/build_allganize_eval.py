"""Build the Allganize law/public eval excerpt (phase D1.1).

One-time, DETERMINISTIC builder for the 3rd domain (external Korean gold).
Reads ONLY the lightweight metadata parquets (questions_meta ~102KB,
documents_meta ~9KB) — NOT the 430MB PDF parquet — and writes a small,
self-contained excerpt (data/allganize_eval/allganize_excerpt.jsonl).

This is D1.1 only: eval-set extraction + verification. Corpus indexing (D1.2)
and the EvalSetProvider/adapter (D2) come later and are out of scope here.

Source : datalama/RAG-Evaluation-Dataset-KO (split=test, metadata parquets)
         extended from allganize/RAG-Evaluation-Dataset-KO
License: MIT (see data/allganize_eval/LICENSE.md)

Selection (law 20 + public 20 = 40; each domain paragraph 12 + table 8,
image/text excluded): round-robin across documents (pid) to maximize doc
diversity, deterministic by (pid number, qid number). No randomness.

Usage: python scripts/build_allganize_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "datalama/RAG-Evaluation-Dataset-KO"
OUT = Path("data/allganize_eval/allganize_excerpt.jsonl")
CORPUS = Path("data/allganize_eval/allganize_corpus.jsonl")
DOMAINS = ["law", "public"]
SLICE_TARGETS = {"paragraph": 12, "table": 8}  # image/text context types excluded


def _text_bearing_pids() -> set[str] | None:
    """Documents with at least one extractable-text page (built by D1.2).

    A few source PDFs are fully scanned images (0 extractable text) — a question
    whose gold DOCUMENT is one of those would be an unavoidable retrieval_miss
    regardless of retriever quality, distorting the baseline. If the corpus exists
    we exclude such documents from selection (honest); if not, return None (no filter)."""
    if not CORPUS.exists():
        return None
    pids: set[str] = set()
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("text", "").strip():
            pids.add(r["pid"])
    return pids


def _num(s: str) -> int:
    """Sort key for 'q_12' / 'p_7' -> 12 / 7 (deterministic numeric order)."""
    return int(s.rsplit("_", 1)[-1])


def select(cands: pd.DataFrame, n: int) -> list[dict]:
    """Round-robin across pid (document) to spread picks over many docs."""
    by_pid: dict[str, list] = {}
    for rec in cands.to_dict("records"):
        by_pid.setdefault(rec["pid"], []).append(rec)
    for pid in by_pid:
        by_pid[pid].sort(key=lambda r: _num(r["qid"]))
    pids = sorted(by_pid, key=_num)
    picked: list[dict] = []
    round_i = 0
    while len(picked) < n:
        progressed = False
        for pid in pids:
            if round_i < len(by_pid[pid]):
                picked.append(by_pid[pid][round_i])
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:  # exhausted all docs before reaching n
            break
        round_i += 1
    return picked


def main() -> int:
    q = hf_hub_download(REPO, "questions_meta/meta-00000-of-00001.parquet", repo_type="dataset")
    d = hf_hub_download(REPO, "documents_meta/meta-00000-of-00001.parquet", repo_type="dataset")
    dfq = pd.read_parquet(q)
    dfd = pd.read_parquet(d)
    corpus_pids = set(dfd["pid"])  # gold-pid existence check is against this

    # domain lives in documents_meta, not questions_meta -> join on pid
    m = dfq.merge(dfd[["pid", "domain"]], on="pid", how="left")

    text_pids = _text_bearing_pids()
    if text_pids is not None:
        excluded = sorted(set(dfd[dfd["domain"].isin(DOMAINS)]["pid"]) - text_pids, key=_num)
        print(f"image-only documents excluded from selection: {excluded or 'none'}")

    rows: list[dict] = []
    seen_qid: set[str] = set()
    for domain in DOMAINS:
        for ctype, n in SLICE_TARGETS.items():
            cands = m[
                (m["domain"] == domain)
                & (m["context_type"] == ctype)
                & (m["target_answer"].str.strip() != "")
            ]
            if text_pids is not None:
                cands = cands[cands["pid"].isin(text_pids)]
            for rec in select(cands, n):
                if rec["qid"] in seen_qid:  # dedupe (qids are unique, but guard)
                    continue
                seen_qid.add(rec["qid"])
                rows.append({
                    "qid": rec["qid"],
                    "pid": rec["pid"],              # gold document id
                    "file_name": rec["file_name"],
                    "domain": domain,
                    "context_type": ctype,          # -> EvalCase.slice
                    "target_page_no": str(rec["target_page_no"]),
                    "question": rec["question"],
                    "target_answer": rec["target_answer"],  # -> TextAnswer.key_points
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    # ---- exhaustive verification (D1.1 Done requirement) ----
    empties = [r["qid"] for r in rows if not r["target_answer"].strip()]
    missing = [r["qid"] for r in rows if r["pid"] not in corpus_pids]
    dup = len(rows) != len(seen_qid)
    print(f"wrote {len(rows)} cases -> {OUT}")
    print(f"  target_answer non-empty : {len(rows) - len(empties)}/{len(rows)}"
          f"{'  EMPTY:' + str(empties) if empties else '  OK'}")
    print(f"  gold pid in corpus      : {len(rows) - len(missing)}/{len(rows)}"
          f"{'  MISSING:' + str(missing) if missing else '  OK'}")
    print(f"  unique qid              : {'OK' if not dup else 'DUPLICATE'}")

    print("\n== domain x slice distribution ==")
    df = pd.DataFrame(rows)
    print(pd.crosstab(df["domain"], df["context_type"]))
    print("\n== document (pid) diversity ==")
    for domain in DOMAINS:
        ndoc = df[df["domain"] == domain]["pid"].nunique()
        print(f"  {domain}: {len(df[df['domain']==domain])} cases across {ndoc} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
