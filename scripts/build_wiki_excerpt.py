"""Build the bundled SQuAD 2.0 dev excerpt for the Wiki mini-domain (G2.1).

One-time builder. Reads the official SQuAD 2.0 dev set and writes a small,
self-contained excerpt (data/wiki_eval/squad2_excerpt.jsonl) so the repo carries
no `datasets` dependency — the gate stays pydantic-only and the eval set is
reproducible at clone time.

Source : SQuAD 2.0 dev (rajpurkar/squad_v2)
         https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
License: CC BY-SA 4.0 (see data/wiki_eval/LICENSE.md)

Selection is DETERMINISTIC (dataset order, no randomness): one qualifying paragraph
per title across the first N titles, then 12 short factoid + 8 adversarial no_answer.

Usage: python scripts/build_wiki_excerpt.py /path/to/dev-v2.0.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("data/wiki_eval/squad2_excerpt.jsonl")
N_DOCS = 10
FACTOID_TARGET = 12
NOANSWER_TARGET = 8
MAX_GOLD_WORDS = 5          # keep factoid answers short
CTX_MIN, CTX_MAX = 300, 900  # readable, clean-to-index paragraph length


def qualifying_paragraph(art: dict):
    """First paragraph of an article with >=2 short factoids + >=1 impossible Q."""
    for pi, para in enumerate(art["paragraphs"]):
        ctx = para["context"]
        if not (CTX_MIN <= len(ctx) <= CTX_MAX):
            continue
        fact = [q for q in para["qas"]
                if not q["is_impossible"] and q["answers"]
                and len(q["answers"][0]["text"].split()) <= MAX_GOLD_WORDS]
        imp = [q for q in para["qas"] if q["is_impossible"]]
        if len(fact) >= 2 and imp:
            return pi, ctx, fact, imp
    return None


def main(src: str) -> int:
    data = json.load(open(src, encoding="utf-8"))["data"]

    docs = []  # (doc_id, title, ctx, factoids, impossibles)
    for art in data:
        if len(docs) >= N_DOCS:
            break
        got = qualifying_paragraph(art)
        if got:
            pi, ctx, fact, imp = got
            docs.append((f"{art['title']}#p{pi}", art["title"], ctx, fact, imp))

    # greedy fill to targets: round-robin factoids, then one no_answer per doc
    rows = []
    fcount = 0
    # pass 1: one factoid per doc; pass 2: a second factoid until target
    for take in (0, 1):
        for doc_id, title, ctx, fact, imp in docs:
            if fcount >= FACTOID_TARGET or take >= len(fact):
                continue
            q = fact[take]
            golds = sorted({a["text"].strip() for a in q["answers"]})
            rows.append({"qid": q["id"], "doc_id": doc_id, "title": title,
                         "context": ctx, "question": q["question"],
                         "answers": golds, "is_impossible": False})
            fcount += 1

    ncount = 0
    for doc_id, title, ctx, fact, imp in docs:
        if ncount >= NOANSWER_TARGET:
            break
        q = imp[0]
        rows.append({"qid": q["id"], "doc_id": doc_id, "title": title,
                     "context": ctx, "question": q["question"],
                     "answers": [], "is_impossible": True})
        ncount += 1

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"docs={len(docs)}  factoid={fcount}  no_answer={ncount}  total={len(rows)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dev-v2.0.json"))
