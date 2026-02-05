from typing import Dict, Any, List
from collections import defaultdict
import copy
from .failure_query_tools import query_failure_kb_by_chunks
from pathlib import Path

SOURCE_WEIGHT = {
    "normal": 1.0,
    "drop_ee": 0.85,
    "fuzzy": 0.6,
}

MULTI_HIT_BONUS = 0.1   # bonus per extra hit

def _normalize_results(
    res: Dict[str, Any],
    source_tag: str,
) -> List[Dict[str, Any]]:
    """
    Normalize query_failure_kb_by_chunks output to flat rows
    """
    rows = []
    for r in res.get("merged", []) or []:
        cid = r.get("cause_id")
        if not cid:
            continue
        rows.append({
            "cause_id": str(cid),
            "score": float(r.get("score", 0.0)),
            "source": source_tag,
            "hits": r.get("hits", []),
        })
    return rows

def _calc_final_score(rows):
    score = 0.0
    sources = set()
    for r in rows:
        w = SOURCE_WEIGHT.get(r["source"], 1.0)
        score += r["score"] * w
        sources.add(r["source"])

    # bonus only if multiple retrieval strategies hit
    if len(sources) > 1:
        score *= (1.0 + MULTI_HIT_BONUS * (len(sources) - 1))

    return score

def _build_entity_drop_ee(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove failure_element & failure_effect
    """
    new_entity = copy.deepcopy(entity)
    new_entity["failure_element"] = None
    new_entity["failure_effect"] = None
    return new_entity

def _build_entity_fuzzy(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fuzzy entity: keep only mode & cause
    """
    return {
        "failure_mode": entity.get("failure_mode"),
        "failure_element": None,
        "failure_effect": None,
        "failure_cause": entity.get("failure_cause"),
    }


def query_failure_kb_multiple_retrieval(
    *,
    persist_dir: str,
    entity: Dict[str, Any],
    n_results_each: int = 15,
    fuzzy_multiplier: int = 2,
    # ---- optional filters ----
    source_type=None,
    productPnID=None,
    product_domain=None,
    fmea_type=None,
    system=None,
    discipline=None,
    extra_where=None,
) -> Dict[str, Any]:
    """
    Multiple retrieval orchestrator for FMEA failure KB

    Returns:
    {
        "final_ranked": [
            {
                "failure_id": "...",
                "final_score": float,
                "hits": [...],
            }
        ],
        "debug": {
            "normal": raw,
            "drop_ee": raw,
            "fuzzy": raw,
        }
    }
    """

    # --------------------------
    # 1. Normal retrieval
    # --------------------------
    res_normal = query_failure_kb_by_chunks(
        persist_dir=persist_dir,
        entity=entity,
        n_results_each=n_results_each,
        source_type=source_type,
        productPnID=productPnID,
        product_domain=product_domain,
        fmea_type=fmea_type,
        system=system,
        discipline=discipline,
        extra_where=extra_where,
    )

    # --------------------------
    # 2. Drop element & effect
    # --------------------------
    entity_drop_ee = _build_entity_drop_ee(entity)

    res_drop_ee = query_failure_kb_by_chunks(
        persist_dir=persist_dir,
        entity=entity_drop_ee,
        n_results_each=n_results_each,
        source_type=source_type,
        productPnID=productPnID,
        product_domain=product_domain,
        fmea_type=fmea_type,
        system=system,
        discipline=discipline,
        extra_where=extra_where,
    )

    # --------------------------
    # 3. Fuzzy retrieval
    # --------------------------
    # entity_fuzzy = _build_entity_fuzzy(entity)

    # res_fuzzy = query_failure_kb_by_chunks(
    #     persist_dir=persist_dir,
    #     entity=entity_fuzzy,
    #     n_results_each=n_results_each * fuzzy_multiplier,
    #     source_type=source_type,
    #     productPnID=productPnID,
    #     product_domain=product_domain,
    #     fmea_type=fmea_type,
    #     system=system,
    #     discipline=discipline,
    #     extra_where=extra_where,
    # )

    # --------------------------
    # 4. Normalize & merge
    # --------------------------
    all_rows = []
    all_rows += _normalize_results(res_normal, "normal")
    all_rows += _normalize_results(res_drop_ee, "drop_ee")
    # all_rows += _normalize_results(res_fuzzy, "fuzzy")

    bucket = defaultdict(list)
    for r in all_rows:
        bucket[r["cause_id"]].append(r)

    # --------------------------
    # 5. Final ranking
    # --------------------------
    final_ranked = []
    for cid, rows in bucket.items():
        final_ranked.append({
            "cause_id": cid,
            "final_score": _calc_final_score(rows),
            "hits": rows,   # explainability
        })

    final_ranked.sort(key=lambda x: x["final_score"], reverse=True)

    return {
        "final_ranked": final_ranked,
        "debug": {
            "normal": res_normal,
            "drop_ee": res_drop_ee,
            # "fuzzy": res_fuzzy,
        }
    }

if  __name__ == "__main__":

    KB_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb")

    FAILURE_ENTITY = {
      "failure_mode": "Connection damage from handling",
      "failure_element": "",
      "failure_effect": "Motor not running",
      "failure_cause": "Excess force on connection cables"
    }
    out = query_failure_kb_multiple_retrieval(
        persist_dir=KB_PATH,
        entity=FAILURE_ENTITY,
        n_results_each=15,
    )

    top_failures = out["final_ranked"][:5]
    print(top_failures)
