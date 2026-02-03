from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List, Union

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

from typing import Optional, Dict, Any, List, Union
from collections import defaultdict

from .kb_structure_8d import FailureKB
from .kb_structure_fmea import FMEAFailureKB

def _get_collection(
    persist_dir: Union[str, Path],
    collection_name: str = "all_failure_kb",
):
    client = chromadb.PersistentClient(path=str(persist_dir))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
    )


def _build_where(
    failure_id=None,
    cause_id=None,
    role=None,
    source_type=None,
    fmea_type=None,
    productPnID=None,
    product_domain=None,
    system=None,
    discipline=None,
    extra_where=None,
):
    def clause(k, v):
        if v is None:
            return None
        if isinstance(v, list):
            return {k: {"$in": v}}
        return {k: v}

    clauses = []
    for k, v in [
        ("failure_id", failure_id),
        ("cause_id", cause_id),
        ("role", role),
        ("source_type", source_type),
        ("fmea_type", fmea_type),
        ("productPnID", productPnID),
        ("product_domain", product_domain),
        ("system", system),
        ("discipline", discipline),
    ]:
        c = clause(k, v)
        if c:
            clauses.append(c)

    if extra_where:
        for k, v in extra_where.items():
            c = clause(k, v)
            if c:
                clauses.append(c)

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

# =========================================================
# 1) Semantic search
# =========================================================
def query_failure_kb(
    persist_dir: Union[str, Path],
    query_text: str,
    n_results: int = 10,
    collection_name: str = "all_failure_kb",
    # filters:
    failure_id: Optional[str] = None,
    cause_id: Optional[str] = None,
    role: Optional[Union[str, List[str]]] = None,
    source_type: Optional[Union[str, List[str]]] = None,
    fmea_type: Optional[Union[str, List[str]]] = None,
    productPnID: Optional[Union[str, List[str]]] = None,
    product_domain: Optional[Union[str, List[str]]] = None,
    system: Optional[Union[str, List[str]]] = None,
    discipline: Optional[Union[str, List[str]]] = None,
    extra_where: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
):
    """
    semantic filter: query_text + where filter
    """
    col = _get_collection(persist_dir, collection_name=collection_name)

    where = _build_where(
        failure_id=failure_id,
        cause_id=cause_id,
        role=role,
        source_type=source_type,
        fmea_type=fmea_type,
        productPnID=productPnID,
        product_domain=product_domain,
        system=system,
        discipline=discipline,
        extra_where=extra_where,
    )

    if include is None:
        include = ["documents", "metadatas", "distances"]

    # if where is empty, return None
    where_arg = where if where else None

    return col.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_arg,
        include=include,
    )


# =========================================================
# 2) Query by metadata (group results by metadata)
# =========================================================
def get_by_metadata(
    persist_dir: Union[str, Path],
    collection_name: str = "all_failure_kb",
    limit: int = 1000,
    offset: int = 0,
    # filters:
    failure_id: Optional[str] = None,
    cause_id: Optional[str] = None,
    role: Optional[Union[str, List[str]]] = None,
    source_type: Optional[Union[str, List[str]]] = None,
    fmea_type: Optional[Union[str, List[str]]] = None,
    productPnID: Optional[Union[str, List[str]]] = None,
    product_domain: Optional[Union[str, List[str]]] = None,
    system: Optional[Union[str, List[str]]] = None,
    discipline: Optional[Union[str, List[str]]] = None,
    extra_where: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
):
    """
    Meta filter
    """
    col = _get_collection(persist_dir, collection_name=collection_name)

    where = _build_where(
        failure_id=failure_id,
        cause_id=cause_id,
        role=role,
        source_type=source_type,
        fmea_type=fmea_type,
        productPnID=productPnID,
        product_domain=product_domain,
        system=system,
        discipline=discipline,
        extra_where=extra_where,
    )

    if include is None:
        include = ["documents", "metadatas"]

    where_arg = where if where else None

    return col.get(
        where=where_arg,
        limit=limit,
        offset=offset,
        include=include,
    )


# =========================================================
# 3) Query by ID + metadata
# =========================================================
def get_by_ids(
    persist_dir: Union[str, Path],
    ids: List[str],
    collection_name: str = "all_failure_kb",
    include: Optional[List[str]] = None,
):
    col = _get_collection(persist_dir, collection_name=collection_name)
    if include is None:
        include = ["documents", "metadatas"]
    return col.get(ids=ids, include=include)


def query_failure_kb_by_chunks(
    persist_dir,
    entity: Dict[str, Optional[str]],
    n_results_each: int = 5,
    # Meta filter
    source_type: Optional[Union[str, List[str]]] = None,
    productPnID: Optional[Union[str, List[str]]] = None,
    product_domain: Optional[Union[str, List[str]]] = None,
    fmea_type: Optional[Union[str, List[str]]] = None,
    system: Optional[Union[str, List[str]]] = None,
    discipline: Optional[Union[str, List[str]]] = None,
    extra_where: Optional[Dict[str, Any]] = None,
    # top_n_merged = 3
) -> Dict[str, Any]:
    """
    chunks input:
    {
      "failure_mode": "...",
      "failure_element": "...",
      "failure_effect": "...",
      "failure_cause": "..."
    }

    return:
    {
      "by_role": {role: chroma_query_result, ...},
      "merged": [ {failure_id, score, hits:[...]} , ... ]  # 按 failure_id 聚合
    }
    """
    ROLE_WEIGHT = {
        # failure-level broadcast weights
        "failure_effect": 0.5,
        "failure_element": 0.5,
        "failure_mode": 1.00,   # 

        # direct cause hit weight
        "failure_cause": 1.00,
    }

    failure_kb_FMEA = FMEAFailureKB(persist_dir=persist_dir)
    failure_kb_8d = FailureKB(persist_dir=persist_dir)
    # 1) Speperate role query
    by_role = {}
    for role, text in entity.items():
        if not text or not str(text).strip():
            continue

        by_role[role] = query_failure_kb(
            persist_dir=persist_dir,
            query_text=text,
            role=role,                    
            n_results=n_results_each,
            source_type=source_type,
            productPnID=productPnID,
            product_domain=product_domain,
            fmea_type=fmea_type,
            system=system,
            discipline=discipline,
            extra_where=extra_where,
            include=["documents", "metadatas", "distances"],
        )

    # 2) Aggregation：through failure ids
    #   Simple method: distance：score = sum(1/(1+dist))
    agg = defaultdict(lambda: {
        "failure_id": None,
        "cause_id": None,
        "score": 0.0,
        "source_type": None, 
        "hits": []
    })
    for role, r in by_role.items():
        ids0 = r.get("ids", [[]])[0]
        docs0 = r.get("documents", [[]])[0]
        metas0 = r.get("metadatas", [[]])[0]
        dists0 = r.get("distances", [[]])[0]

        for rid, doc, meta, dist in zip(ids0, docs0, metas0, dists0):
            role_hit = meta.get("role")
            fid = meta.get("failure_id")
            cid = meta.get("cause_id")
            base = 1.0 / (1.0 + float(dist))

            # ---------- case 1: 命中 failure_cause ----------
            if role_hit == "failure_cause" and cid:
                w = ROLE_WEIGHT["failure_cause"]
                a = agg[cid]
                a["failure_id"] = fid
                a["cause_id"] = cid
                a["source_type"] = meta.get("source_type") # For sentences query
                a["score"] += w * base    # 强权重
                a["hits"].append({
                    "from_role": role,
                    "matched_role": role_hit,
                    "weight": w,
                    "distance": dist,
                    "text": doc,
                })


            # ---------- case 2: 命中 failure-level ----------
            elif role_hit in {"failure_effect","failure_mode","failure_element",}:
                w = ROLE_WEIGHT.get(role_hit, 0.5)
                if meta.get("source_type") == "8D": failure_kb = failure_kb_8d
                else: failure_kb = failure_kb_FMEA
                cause_ids = failure_kb.store.get(fid, {}).get("cause_ids", [])
                for c in cause_ids:
                    cid2 = c["cause_id"]
                    a = agg[cid2]
                    a["failure_id"] = fid
                    a["cause_id"] = cid2
                    a["score"] += w * base   # 弱权重（很重要）
                    a["source_type"] = meta.get("source_type")
                    a["hits"].append({
                        "from_role": role,
                        "matched_role": role_hit,
                        "weight": w,
                        "distance": dist,
                        "text": doc,
                    })
    merged = sorted(agg.values(), key=lambda x: x["score"], reverse=True)
    # if top_n_merged is not None:
    #     merged = merged[: top_n_merged]

    return {"by_role": by_role, "merged": merged}


if  __name__ == "__main__":
    KB_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb")

#     res = get_by_metadata(
#     persist_dir=KB_PATH,
#     productPnID=213175,
#     source_type="8D",
#     limit=10,
# )
    
#     res = get_by_ids(
#     persist_dir=KB_PATH,
#     ids=["8D6298110111R01_F1_C1",],
# )
#     print(res)

    # res = query_failure_kb(
    #     persist_dir=KB_PATH,
    #     query_text="DC PCB",
    #     role=["failure_mode", "failure_effect", "failure_cause",],
    #     n_results=3,
    # )

    # Print results
    # for i, (rid, doc, meta, dist) in enumerate(
    #     zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]),
    #     start=1
    # ):
    #     print(f"#{i} id={rid} dist={dist}")
    #     print(f"   role={meta.get('role')} failure_id={meta.get('failure_id')} source={meta.get('source_type')} product={meta.get('productPnID')}")
    #     print(f"   text={doc}")

    FAILURE_ENTITY = {
        "failure_mode": "Relay cannot close",
        # "failure_element": "Motor control",
        # "failure_effect": "Motor cannot start",
        "failure_cause": "Overvoltage due to motor disconnect",
    }

    out = query_failure_kb_by_chunks(
        persist_dir=KB_PATH,
        entity=FAILURE_ENTITY,
        n_results_each=15,  # 可选过滤
        # productPnID=213175,                    # 可选过滤
    )
        # 1) Print Top-k similar results in each failure key
    for role, res in out["by_role"].items():
        print("\n" + "=" * 80)
        print(f"[ROLE QUERY] {role}")
        ids0   = res.get("ids", [[]])[0]
        docs0  = res.get("documents", [[]])[0]
        metas0 = res.get("metadatas", [[]])[0]
        dists0 = res.get("distances", [[]])[0]

        for i, (rid, doc, meta, dist) in enumerate(zip(ids0, docs0, metas0, dists0), start=1):
            print(f"  #{i} id={rid} dist={float(dist):.4f}")
            print(f"     failure_id={meta.get('failure_id')} matched_role={meta.get('role')} "
                f"source={meta.get('source_type')} pn={meta.get('productPnID')}")
            print(f"     text={doc}")

    # 2) Final aggregated result
    print("\n" + "#" * 80)
    print("[MERGED CAUSE RANKING]")

    for rank, item in enumerate(out["merged"][:10], 1):
        print(
            f"\n[{rank}] "
            f"failure_id={item['failure_id']} "
            f"cause_id={item['cause_id']} "
            f"score={item['score']:.4f} "
            # f"direct_hit={item['has_direct_cause_hit']}"
        )

        for h in item["hits"]:
            print(
                f"   - from={h['from_role']} "
                f"matched_role={h['matched_role']} "
                f"dist={h['distance']:.4f} "
                f"weight={h['weight']}"
            )
            print(f"     {h['text']}")
        