from __future__ import annotations

from .failure_query_tools import query_failure_kb_by_chunks
from .sentence_query_tools import query_sentence_kb_by_chunks

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

def _format_sentence_by_role(by_role: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Output each failure key's top-k similar sentence_id + text (and distance).
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for role, r in (by_role or {}).items():
        ids0 = r.get("ids", [[]])[0]
        docs0 = r.get("documents", [[]])[0]
        dists0 = r.get("distances", [[]])[0]
        metas0 = r.get("metadatas", [[]])[0]

        items = []
        for sid, doc, dist, meta in zip(ids0, docs0, dists0, metas0):
            items.append(
                {
                    "sentence_id": sid,
                    "text": doc,
                    "distance": float(dist),
                    "metadata": meta,
                }
            )
        out[role] = items
    return out

def _format_sentence_merged(merged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Output aggregated sentence hits per group_id with sentence_id + text.
    """
    out: List[Dict[str, Any]] = []
    for m in merged or []:
        out.append(
            {
                "group_id": m.get("group_id"),
                "score": float(m.get("score", 0.0)),
                "sentences": [
                    {
                        "sentence_id": h.get("sentence_id"),
                        "text": h.get("text"),
                        "chunk": h.get("from_chunk") or h.get("chunk"),
                        "distance": float(h.get("distance", 0.0)),
                        "metadata": h.get("metadata", {}),
                    }
                    for h in (m.get("hits") or [])
                ],
            }
        )
    return out

def save_json(out: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def query_fmea_top_failures_with_sentences(
    failure_kb_dir,
    sentence_kb_dir,
    entity: dict,
    *,
    top_n_failures: int = 10,
    n_results_each: int = 5,     # failure kb 每个chunk查多少
    sentence_top_k: int = 3,     # sentence kb 每个chunk top-k
    # ---- failure kb filters ----
    source_type=None,
    productPnID=None,
    product_domain=None,
    fmea_type=None,
    system=None,
    discipline=None,
    extra_where_failure=None,
    # ---- sentence kb filters ----
    case_id=None,
    subject=None,
    status=None,
    faithful_score=None,
    extra_where_sentence=None,
):

     # 1) Failure KB query
    failure_res = query_failure_kb_by_chunks(
        persist_dir=failure_kb_dir,
        entity=entity,
        n_results_each=n_results_each,
        source_type=source_type,
        productPnID=productPnID,
        product_domain=product_domain,
        fmea_type=fmea_type,
        system=system,
        discipline=discipline,
        extra_where=extra_where_failure,
    )
    merged_failures = (failure_res.get("merged") or [])[:top_n_failures]
    results = []

    for f in merged_failures:
        cause_id = f.get("cause_id") 
        failure_id = f.get("failure_id")
        source_type = f.get("source_type")
        score = float(f.get("score", 0.0))
        hits = f.get("hits") or []

        is_8d = (f.get("source_type") == "8D")

        role_sum = defaultdict(float)
        best_by_role = {}  # role -> {text, distance}

        for h in hits:
            role = h.get("from_role") or h.get("role_queried")
            text = h.get("text")
            dist = h.get("distance")
            w = float(h.get("weight", 1.0))

            if not role or dist is None:
                continue

            dist = float(dist)
            base = 1.0 / (1.0 + dist)
            contrib = w * base
            role_sum[role] += contrib

            # 每个 role 保存一条 text：取 distance 最小的那条
            if text:
                if role not in best_by_role or dist < best_by_role[role]["distance"]:
                    best_by_role[role] = {"text": text, "distance": dist}

        # 输出：每个 role 下带 score + text
        out_item = {
            "cause_id": cause_id,
            "failure_id": failure_id,
            "source_type":source_type ,
            "score": score,
            "by_role": {
                role: {
                    "score": float(role_sum[role]),
                    "text": best_by_role.get(role, {}).get("text"),
                    "distance": best_by_role.get(role, {}).get("distance"),
                }
                for role in role_sum.keys()
            }
        }

        # 4) 
        if is_8d and failure_id:
            sentence_res = query_sentence_kb_by_chunks(
                persist_dir=sentence_kb_dir,
                entity=entity,
                n_results_each=sentence_top_k,
                failure_id=failure_id,
                case_id=case_id,
                subject=subject,
                status=status,
                faithful_score=faithful_score,
                productPnID=productPnID,
                product_domain=product_domain,
                extra_where=extra_where_sentence,
            )

            sentences_by_role = {}
            for role, r in (sentence_res.get("by_role") or {}).items():
                ids0 = r.get("ids", [[]])[0]
                docs0 = r.get("documents", [[]])[0]
                dists0 = r.get("distances", [[]])[0]

                sentences_by_role[role] = [
                    {
                        "sentence_id": sid,
                        "text": doc,
                        "distance": float(dist),
                    }
                    for sid, doc, dist in zip(ids0, docs0, dists0)
                ]

            out_item["sentences_by_role"] = sentences_by_role

        results.append(out_item)

    return {"results": results}

if __name__ == "__main__":
    SENTENCE_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\sentence_kb")
    FAILURE_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb")

    FAILURE_ENTITY = {
    "failure_mode": "Capacitor short failure",
    "failure_element": "Power electronics",
    "failure_effect": "Motor control lost",
    "failure_cause": "Capacitor mechanically stressed"
    }
    out = query_fmea_top_failures_with_sentences(
    failure_kb_dir=FAILURE_PATH,
    sentence_kb_dir=SENTENCE_PATH,
    entity=FAILURE_ENTITY,
    top_n_failures = 10,
    n_results_each=10,
    sentence_top_k=3,
    product_domain="motor_drives",
    # source_type="8D"
)
    save_json(out, "out.json")