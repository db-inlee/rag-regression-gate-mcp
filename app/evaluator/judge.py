"""LLM-as-judge for body_text answers (Phase 2.2).

Scores a free-text answer against the reference key_points for semantic
sufficiency. JSON schema is enforced; a parse failure degrades gracefully to an
(incorrect, parse_ok=False) verdict so one bad case never breaks the whole run.
The judge model id + the API-reported version are recorded on every verdict.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.config import RagConfig

_SYSTEM = (
    "당신은 엄격한 채점자다. 질문과 '참조 핵심 포인트'(정답이 담아야 할 사실)와 "
    "후보 답이 주어진다. 후보 답이 핵심 포인트를 의미상 충분히 담고 사실과 일치하면 "
    "correct=true, 핵심이 빠졌거나 사실(숫자·대상·방향 등)이 틀리면 correct=false로 판정하라. "
    '표현이 달라도 의미가 맞으면 correct=true. 반드시 JSON으로만 답하라: '
    '{"correct": true/false, "reason": "간단한 근거"}'
)


class JudgeVerdict(BaseModel):
    correct: bool
    reason: str


def _parse_verdict(raw: str) -> tuple[dict, bool]:
    """(verdict, parse_ok). Graceful fallback on malformed output."""
    try:
        v = JudgeVerdict.model_validate_json(raw)
        return {"correct": v.correct, "reason": v.reason}, True
    except Exception:  # noqa: BLE001 - never let a bad judge output crash the run
        return {"correct": False, "reason": f"parse_failed: {(raw or '')[:120]!r}"}, False


def judge_body_text(question: str, key_points: list[str], answer: str | None,
                    config: RagConfig, client=None) -> dict:
    from openai import OpenAI

    if not (answer or "").strip():
        return {"correct": False, "reason": "empty answer", "parse_ok": True,
                "judge_model": config.judge_model, "judge_version": None}

    client = client or OpenAI()
    user = (
        f"질문: {question}\n"
        f"참조 핵심 포인트: {key_points}\n"
        f"후보 답: {answer}\n"
        "위 후보 답이 핵심 포인트를 의미상 충분히 담는가?"
    )
    resp = client.chat.completions.create(
        model=config.judge_model,
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    verdict, parse_ok = _parse_verdict(resp.choices[0].message.content)
    return {**verdict, "parse_ok": parse_ok,
            "judge_model": config.judge_model, "judge_version": resp.model}
