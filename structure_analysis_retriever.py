"""
Structure Analysis → Role-aware Retrieval (+ optional Cross-Encoder rerank) → Failure-level Aggregation (KB by_role) →
Chain-aware Rerank (coverage + evidence + stability + confidence) → Debug Trace (console + optional JSONL)

Key upgrades vs your previous version:
1) Aggregation is KB-centric: uses retriever output fields `by_role` and `sentences_by_role` (NOT query role).
2) Debug/trace: prints each query + top retriever hits (with by_role), and final top chains.
3) Optional: keep / skip cross-encoder rerank_failures (toggle USE_CROSS_ENCODER).
4) Robust to missing fields and supports multiple roles per hit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from Retriever.integration_tools import query_fmea_top_failures_with_sentences
from Reranker.cross_encoder import rerank_failures


# =====================================================
# 0) Config
# =====================================================

USE_CROSS_ENCODER = True         # set False if you want retriever-only + chain rerank
DEBUG = True
DEBUG_TOP_RETRIEVER = 5           # print top-N retriever hits per query
DEBUG_TOP_FINAL = 20              # print top-N final chains
TRACE_JSONL_PATH: Optional[str] = None  # e.g. "./trace.jsonl" to save per-query trace


# Retriver output uses KB role names: failure_mode, root_cause, failure_effect, failure_element, etc.
# We'll map them to canonical roles used in chain scoring.
KB_ROLE_TO_CANON = {
    "failure_mode": "mode",
    "failure_cause": "cause",
    "failure_effect": "effect",
    "failure_element": "element",
}

QUERY_ROLE_WEIGHT = {
    "mode+cause": 2.0,
    "mode": 1.0,
    "cause": 1.0,
    "effect": 0.5
}


# =====================================================
# 1) Helpers
# =====================================================

def _safe_dump(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def _write_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _confidence_weight(conf: Optional[str]) -> float:
    m = {"high": 1.0, "medium": 0.6, "low": 0.3}
    return m.get((conf or "").lower(), 0.5)


def _canon_role(kb_role: str) -> str:
    return KB_ROLE_TO_CANON.get(kb_role, kb_role)


def to_jsonable(obj):
    """Recursively convert non-JSON-serializable objects to JSONable ones."""
    # set -> list
    if isinstance(obj, set):
        return sorted(list(obj))
    # defaultdict -> dict
    if isinstance(obj, defaultdict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    # dict
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    # Path
    if isinstance(obj, Path):
        return str(obj)
    # other primitives
    return obj

def extract_ids_from_rerank_item(rr: Dict[str, Any]):
    # case 1: retriever raw
    if "failure_id" in rr or "cause_id" in rr:
        return rr.get("failure_id"), rr.get("cause_id")

    # case 2: cross encoder rerank
    f = rr.get("failure") or {}
    return f.get("failure_id"), f.get("cause_id")


# =====================================================
# 2) Structure → Role-aware Queries (query-centric, for retrieval only)
# =====================================================

def expand_structure_to_role_queries(
    structure_nodes: List[Dict[str, Any]],
    max_mode_cause_pairs: int = 6
) -> List[Dict[str, Any]]:
    """
    Generate:
      - mode-only queries
      - cause-only queries
      - mode+cause combined queries (limited)
    """
    queries: List[Dict[str, Any]] = []

    for node in structure_nodes:
        element = node.get("failure_element") or node.get("element") or ""
        element_id = node.get("element_id") or node.get("id") or ""

        modes = node.get("modes") or node.get("failure_modes") or []
        causes = node.get("causes") or node.get("failure_causes") or []
        effects = node.get("effects") or node.get("failure_effects") or []

        # --- mode-only ---
        for m in modes:
            queries.append({
                "element_id": element_id,
                "query_role": "mode",
                "entity": {
                    "failure_element": element,
                    "failure_mode": m
                }
            })

        # --- cause-only ---
        for c in causes:
            queries.append({
                "element_id": element_id,
                "query_role": "cause",
                "entity": {
                    "failure_element": element,
                    "failure_cause": c
                }
            })

        # --- mode + cause (controlled combination) ---
        pairs = []
        for m in modes:
            for c in causes:
                pairs.append((m, c))

        # Limited number
        for m, c in pairs[:max_mode_cause_pairs]:
            queries.append({
                "element_id": element_id,
                "query_role": "mode+cause",
                "entity": {
                    "failure_element": element,
                    "failure_mode": m,
                    "failure_cause": c
                }
            })

        # # --- effect-only（可选，保留） ---
        # for e in effects:
        #     queries.append({
        #         "element_id": element_id,
        #         "query_role": "effect",
        #         "entity": {
        #             "failure_element": element,
        #             "failure_effect": e
        #         }
        #     })

    return queries



# =====================================================
# 3) Retrieve (and optionally Cross-Encoder rerank) for a single query
# =====================================================

def retrieve_one(
    entity: Dict[str, Any],
    failure_kb_dir: Path,
    sentence_kb_dir: Path,
    product_domain: Optional[str],
    top_n_failures: int,
    n_results_each: int,
    sentence_top_k: int,
) -> Dict[str, Any]:
    """Call your retriever. Return full retriever_result dict."""
    return query_fmea_top_failures_with_sentences(
        failure_kb_dir=failure_kb_dir,
        sentence_kb_dir=sentence_kb_dir,
        entity=entity,
        top_n_failures=top_n_failures,
        n_results_each=n_results_each,
        sentence_top_k=sentence_top_k,
        product_domain=product_domain
    )


def maybe_cross_encode_rerank(
    entity: Dict[str, Any],
    retriever_results: List[Dict[str, Any]],
    ed_store: Dict[str, Any],
    fmea_store: Dict[str, Any],
    top_k: int
) -> List[Dict[str, Any]]:
    """
    If USE_CROSS_ENCODER is True, call rerank_failures to reorder / score results.
    Expect rerank_failures returns list items containing at least failure_id and score (your current behavior).
    """
    if not USE_CROSS_ENCODER:
        # Normalize to a common structure.
        # retriever_results already contain failure_id and score; keep top_k.
        out = []
        for r in retriever_results[:top_k]:
            out.append({
                "failure_id": r.get("failure_id"),
                "cause_id": r.get("cause_id"),
                "score": float(r.get("score", 0.0)),
                "_raw": r
            })
        return out

    reranked = rerank_failures(
        query_entity=entity,
        retriever_results=retriever_results,
        fmea_8d_json=ed_store,
        fmea_json=fmea_store,
        top_k=top_k
    )
    return reranked


# =====================================================
# 4) Failure-level aggregation (KB-centric using by_role / sentences_by_role)
# =====================================================

def _init_agg() -> Dict[str, Any]:
    """
    Aggregation record per failure_id:
      element_ids: set of element_id that hit this failure
      roles_hit: set of canonical roles (mode/cause/effect/element)
      kb_roles_hit: set of kb roles (failure_mode/root_cause/...)
      role_scores: canon_role -> list[float] (from by_role scores if present, else fall back)
      role_sentence_count: canon_role -> int (count of sentences in sentences_by_role)
      query_supports: list of which query (element_id + query_role + query_text) supported this failure
      meta: source_type/confidence/released_year/productPnID
    """
    return {
        "element_ids": set(),
        "roles_hit": set(),
        "kb_roles_hit": set(),
        "role_scores": defaultdict(list),
        "role_sentence_count": defaultdict(int),
        "query_supports": [],
        "meta": {
            "source_type": None,
            "confidence": None,
            "released_year": None,
            "productPnID": None,
        }
    }


def aggregate_from_retriever_item(
    agg,
    element_id,
    query_role,
    entity,
    retr_item,
    fallback_score
):
    cause_id = retr_item.get("cause_id")
    if not cause_id:
        return

    rec = agg.setdefault(cause_id, {
        "cause_id": cause_id,
        "failure_id": retr_item.get("failure_id"),
        "element_ids": set(),
        "roles_hit": set(),
        "scores": defaultdict(list),
        "sentence_count": defaultdict(int),
        "meta": {
            "source_type": retr_item.get("source_type"),
            "confidence": retr_item.get("confidence"),
            "released_year": retr_item.get("released_year"),
        }
    })

    rec["element_ids"].add(element_id)

    # === KB role-aware aggregation ===
    for kb_role, hit in retr_item.get("by_role", {}).items():
        if kb_role == "failure_element":
            continue

        # 映射 role
        if kb_role == "failure_mode":
            role = "failure_mode"
        elif kb_role == "failure_cause":
            role = "failure_cause"
        elif kb_role == "failure_effect":
            role = "failure_effect"
        else:
            continue

        rec["roles_hit"].add(role)
        weighted_score = fallback_score * QUERY_ROLE_WEIGHT[query_role]
        rec["scores"][role].append(weighted_score)

    # === sentence evidence ===
    for kb_role, sents in retr_item.get("sentences_by_role", {}).items():
        if kb_role == "failure_mode":
            rec["sentence_count"]["mode"] += len(sents)
        elif kb_role == "failure_cause":
            rec["sentence_count"]["cause"] += len(sents)
        elif kb_role == "failure_effect":
            rec["sentence_count"]["effect"] += len(sents)


# =====================================================
# 5) Chain-aware rerank (failure-centric)
# =====================================================

def rerank_cause_chains(agg):
    ranked = []

    for cid, info in agg.items():
        roles = info["roles_hit"]
        if not roles:
            continue

        role_score = (
            (1.2 if "cause" in roles else 0) +
            (1.0 if "mode" in roles else 0) +
            (0.8 if "effect" in roles else 0)
        )

        evidence_score = sum(
            math.log(1 + info["sentence_count"].get(r, 0))
            for r in roles
        )

        all_scores = []
        for s in info["scores"].values():
            all_scores.extend(s)

        avg_score = sum(all_scores) / max(len(all_scores), 1)

        final_score = (
            0.4 * role_score +
            0.35 * evidence_score +
            0.25 * avg_score
        )

        ranked.append({
            "cause_id": cid,
            "failure_id": info["failure_id"],
            "final_score": round(final_score, 4),
            "roles": list(roles),
            "evidence": info["sentence_count"],
            "meta": info["meta"]
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked


# =====================================================
# 6) Orchestration: Structure → Candidates → Aggregate → Chain Rerank
# =====================================================

def query_structure_fmea(
    structure_input: Dict[str, Any],
    failure_kb_dir: Path,
    sentence_kb_dir: Path,
    ed_store: Dict[str, Any],
    fmea_store: Dict[str, Any],
    top_n_failures: int = 30,
    n_results_each: int = 12,
    sentence_top_k: int = 3,
    top_k: int = 10,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Returns:
      {
        "final_ranked": [...],
        "agg": {...}  # internal aggregation (optional)
      }
    """
    nodes = structure_input.get("nodes") or []
    role_queries = expand_structure_to_role_queries(nodes)

    agg: Dict[str, Any] = {}  # failure_id -> record

    for qi, q in enumerate(role_queries, 1):
        element_id = q["element_id"]
        query_role = q["query_role"]
        entity = q["entity"]

        retriever_result = retrieve_one(
            entity=entity,
            failure_kb_dir=failure_kb_dir,
            sentence_kb_dir=sentence_kb_dir,
            product_domain=structure_input.get("product_domain"),
            top_n_failures=top_n_failures,
            n_results_each=n_results_each,
            sentence_top_k=sentence_top_k
        )

        retriever_results = (
            retriever_result.get("results", [])
            if isinstance(retriever_result, dict)
            else (retriever_result or [])
        )

        # Debug: print query + top retriever hits with KB roles
        if debug:
            print("\n" + "=" * 80)
            print(f"[QUERY {qi}/{len(role_queries)}] element_id={element_id} query_role={query_role}")
            print(_safe_dump(entity))
            print(f"[RETRIEVER TOP {min(DEBUG_TOP_RETRIEVER, len(retriever_results))}]")
            for r in retriever_results[:DEBUG_TOP_RETRIEVER]:
                kb_roles = list((r.get("by_role") or {}).keys())
                print(
                    f"  - failure_id={r.get('failure_id')} cause_id={r.get('cause_id')} "
                    f"score={float(r.get('score', 0.0)):.4f} by_role={kb_roles}"
                )

        # Optional: cross-encoder rerank (query-centric)
        reranked_list = maybe_cross_encode_rerank(
            entity=entity,
            retriever_results=retriever_results,
            ed_store=ed_store,
            fmea_store=fmea_store,
            top_k=top_k
        )

        # For aggregation, we want KB by_role + sentences_by_role.
        # If we used cross-encoder, items may not carry these fields.
        # Solution: merge: keep reranked order/scores, but attach raw retriever item by failure_id/cause_id.
        retr_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in retriever_results:
            fid = r.get("failure_id") or ""
            cid = r.get("cause_id") or ""
            if fid:
                retr_map[(fid, cid)] = r

        # Build list of "items_to_aggregate" with best available raw info
        items_to_aggregate: List[Tuple[Dict[str, Any], float]] = []
        for rr in reranked_list:
            fid, cid = extract_ids_from_rerank_item(rr)

            # score 来源
            score = (
                rr.get("score")
                or rr.get("rerank_score")
                or 0.0
            )
            score = float(score)

            raw = None
            if fid:
                raw = retr_map.get((fid, cid)) or retr_map.get((fid, ""))

            if raw is None:
                # ⚠️ 不能直接用 rr（没有 by_role）
                continue

            items_to_aggregate.append((raw, score))

        # Aggregate
        for raw_item, fallback_score in items_to_aggregate:
            aggregate_from_retriever_item(
                agg=agg,
                element_id=element_id,
                query_role=query_role,
                entity=entity,
                retr_item=raw_item,
                fallback_score=fallback_score
            )

        # Optional trace JSONL
        if TRACE_JSONL_PATH:
            _write_jsonl(TRACE_JSONL_PATH, {
                "query_index": qi,
                "element_id": element_id,
                "query_role": query_role,
                "entity": entity,
                "retriever_top": [
                    {
                        "failure_id": r.get("failure_id"),
                        "cause_id": r.get("cause_id"),
                        "score": r.get("score"),
                        "by_role": list((r.get("by_role") or {}).keys())
                    } for r in retriever_results[:DEBUG_TOP_RETRIEVER]
                ],
                "reranked_top": [
                    {
                        "failure_id": r.get("failure_id"),
                        "cause_id": r.get("cause_id"),
                        "score": r.get("score")
                    } for r in reranked_list[:min(10, len(reranked_list))]
                ]
            })

    # Final chain-aware rerank
    final_ranked = rerank_cause_chains(agg)

    if debug:
        print("\n" + "#" * 80)
        print(f"[FINAL TOP {min(DEBUG_TOP_FINAL, len(final_ranked))}]")
        for i, x in enumerate(final_ranked[:DEBUG_TOP_FINAL], 1):
            print(
                f"[{i}] failure_id={x['failure_id']} final={x['final_score']:.4f} "
                f"chain_roles={x['roles']} evidence={x['evidence']} "
                f"conf={x['meta'].get('confidence')} src={x['meta'].get('source_type')}"
            )

    return {
        "final_ranked": final_ranked,
        "agg": agg  # you can drop this if output too large
    }


# =====================================================
# 7) Example main
# =====================================================

if __name__ == "__main__":

    # --------- update these paths ----------
    ED_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\8d_cause_store.json"
    FMEA_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\fmea_cause_store.json"
    RETRIEVER_RESULT = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\out.json"
    SENTENCE_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\sentence_kb")
    FAILURE_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb")
    OUTPUT = r"./structure_fmea_result.json"
    # --------------------------------------

    with open(ED_JSON, "r", encoding="utf-8") as f:
        ed_store = json.load(f)

    with open(FMEA_JSON, "r", encoding="utf-8") as f:
        fmea_store = json.load(f)

    # Example structure input
    structure_input = {
        "product_domain": "motor_drives",
        "nodes": [
            {
                "element_id": "E1",
                "failure_element": "Motor control",
                "modes": [
                    "Component break-down",
                    "Unbalanced motor currents",
                    "Incorrect interpretation zero-crossing",
                    "No detection",
                    "Welded relay",
                    "Relay cannot close"
                ],
                "causes": [
                    "Cooling insufficient",
                    "Compressor vibrations",
                    "(Starting) Motor current too high for chosen components",
                    "Overvoltage due to motor disconnect",
                    "Under Voltage due to incorrect triggering ",
                    "Live switching of relays",
                    "Priority zero-crossing interrupt too low",
                    "Open loop control",
                    "No (correctly designed) snubber design "
                ],
                "effects": [
                    "Motor cannot start",
                    "Motor starts without soft start",
                    "Overcurrent towards motor",
                ]
            }
        ]
    }

    out = query_structure_fmea(
        structure_input=structure_input,
        failure_kb_dir=FAILURE_PATH,
        sentence_kb_dir=SENTENCE_PATH,
        ed_store=ed_store,
        fmea_store=fmea_store,
        top_n_failures=30,
        n_results_each=12,
        sentence_top_k=3,
        top_k=10,
        debug=DEBUG
    )

    # with open(OUTPUT, "w", encoding="utf-8") as f:
    #     json.dump(out, f, ensure_ascii=False, indent=2)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(out), f, ensure_ascii=False, indent=2)

    print(f"\nSaved → {OUTPUT}")


