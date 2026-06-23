# Test fixtures (synthetic gate inputs)

Synthetic, **reproducible** gate inputs for `tests/test_gate_behavior.py`. They are
derived from `examples/allganize_baseline` (no new RAG runs). Regenerate with:

```bash
python tests/fixtures/make_fixtures.py
```

## `allganize_candidate_1case/`

A copy of `examples/allganize_baseline` (`run.jsonl` + `attribution.jsonl`) with
**exactly one** answerable case flipped `correct → hallucination`:

- `attribution.jsonl`: the first `answerable && correct` record gets
  `correct=false`, `primary_failure="hallucination"`.
- Effect: `answerable_accuracy` drops by 1/40 = 0.025.

**Why**: the baseline `answerable_accuracy` noise band has `std ≈ 0.02`, and the gate
applies a ±1-case floor. A single-case change therefore sits inside the noise band /
at the floor, so the gate must **not FAIL** on it — it returns WARN at most. This
fixture pins the false-alarm guard (`test_noise_band_floor_prevents_false_alarm`,
`test_warn_on_borderline_change`).

> Only `attribution.jsonl` is read by the engine (`detect_paths`); `run.jsonl` is
> copied along to keep the run-dir contract intact.
