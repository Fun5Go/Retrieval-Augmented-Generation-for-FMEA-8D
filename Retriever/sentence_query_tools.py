from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List, Union

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

from typing import Optional, Dict, Any, List, Union
from collections import defaultdict

def _get_sentence_collection(
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

def _build_sentence_where(
    failure_id=None,
    cause_id=None,
    sentence_role=None,
    case_id=None,
    productPnID=None,
    product_domain=None,
    source_section=None,
    status=None,
    subject = None,
    min_faithful_score=None,
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
        ("sentence_role", sentence_role),
        ("case_id", case_id),
        ("productPnID", productPnID),
        ("product_domain", product_domain),
        ("source_section", source_section),
        ("status", status),
        ("subject",subject),
    ]:
        c = clause(k, v)
        if c:
            clauses.append(c)

    if min_faithful_score is not None:
        clauses.append({"faithful_score": {"$gte": int(min_faithful_score)}})

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

def query_sentence_kb(
    persist_dir: Union[str, Path],
    query_text: str,
    n_results: int = 10,
    collection_name: str = "sentences",
    # filters
    failure_id: Optional[str] = None,
    cause_id: Optional[str] = None,
    sentence_role: Optional[Union[str, List[str]]] = None,
    case_id: Optional[str] = None,
    productPnID: Optional[Union[str, List[str]]] = None,
    product_domain: Optional[Union[str, List[str]]] = None,
    source_section: Optional[Union[str, List[str]]] = None,
    status: Optional[Union[str, List[str]]] = None,
    subject: Optional[Union[str, List[str]]] = None,
    min_faithful_score: Optional[int] = None,
    extra_where: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
):
    col = _get_sentence_collection(persist_dir, collection_name)

    where = _build_sentence_where(
        failure_id=failure_id,
        cause_id=cause_id,
        sentence_role=sentence_role,
        case_id=case_id,
        productPnID=productPnID,
        product_domain=product_domain,
        source_section=source_section,
        status=status,
        subject=subject,
        min_faithful_score=min_faithful_score,
        extra_where=extra_where,
    )

    if include is None:
        include = ["documents", "metadatas", "distances"]

    return col.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where if where else None,
        include=include,
    )

def get_sentences_by_metadata(
    persist_dir: Union[str, Path],
    collection_name: str = "sentences",
    limit: int = 1000,
    offset: int = 0,
    # filters
    failure_id: Optional[str] = None,
    cause_id: Optional[str] = None,
    sentence_role: Optional[Union[str, List[str]]] = None,
    case_id: Optional[str] = None,
    productPnID: Optional[Union[str, List[str]]] = None,
    product_domain: Optional[Union[str, List[str]]] = None,
    source_section: Optional[Union[str, List[str]]] = None,
    status: Optional[Union[str, List[str]]] = None,
    subject: Optional[Union[str, List[str]]] = None,
    min_faithful_score: Optional[int] = None,
    extra_where: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
):
    col = _get_sentence_collection(persist_dir, collection_name)

    where = _build_sentence_where(
        failure_id=failure_id,
        cause_id=cause_id,
        sentence_role=sentence_role,
        case_id=case_id,
        productPnID=productPnID,
        product_domain=product_domain,
        source_section=source_section,
        status=status,
        subject = subject,
        min_faithful_score=min_faithful_score,
        extra_where=extra_where,
    )

    if include is None:
        include = ["documents", "metadatas"]

    return col.get(
        where=where if where else None,
        limit=limit,
        offset=offset,
        include=include,
    )


# =========================================================
# Query by IDs
# =========================================================
def get_sentences_by_ids(
    persist_dir: Union[str, Path],
    ids: List[str],
    collection_name: str = "sentences",
    include: Optional[List[str]] = None,
):
    col = _get_sentence_collection(persist_dir, collection_name=collection_name)
    if include is None:
        include = ["documents", "metadatas"]
    return col.get(ids=ids, include=include)


def query_sentence_kb_by_chunks(
    persist_dir: Union[str, Path],
    entity: Dict[str, Optional[str]],
    n_results_each: int = 5,
    collection_name: str = "sentences",
    # meta filters (optional):
    case_id: Optional[Union[str, List[str]]] = None,
    failure_id: Optional[Union[str, List[str]]] = None,
    cause_id: Optional[Union[str, List[str]]] = None,
    source_section: Optional[Union[str, List[str]]] = None,
    status: Optional[Union[str, List[str]]] = None,
    subject: Optional[Union[str, List[str]]] = None,
    faithful_score: Optional[Union[int, List[int], Dict[str, Any]]] = None,
    productPnID: Optional[Union[str, List[str]]] = None,
    product_domain: Optional[Union[str, List[str]]] = None,
    extra_where: Optional[Dict[str, Any]] = None,
    # aggregation:
    group_by: str = "failure_id",  # or "case_id" / "cause_id"
) -> Dict[str, Any]:
    """
    chunks input example:
      {
        "symptom": "...",
        "context": "...",
        "fix": "..."
      }

    For each chunk-key, we treat it as a sentence_role filter by default
    (i.e. query text under sentence_role = chunk-key).
    If you don't want that behavior, remove sentence_role=role in query loop.
    """
    QUERY_SECTION_MAP = {
    "failure_element": ["D2", "D3"],
    "failure_mode": ["D2", "D3"],
    "failure_effect": ["D2", "D3"],
    "failure_cause": ["D3","D4"],
    }

    CHUNK_WEIGHT = {
    "failure_mode": 1.3,
    "failure_cause": 1.5,
    "failure_element": 1.0,
    "failure_effect": 1.0,
}
    by_role = {}

    for role, text in entity.items():
        if not text or not str(text).strip():
            continue

        role_source_section = QUERY_SECTION_MAP.get(role)

        by_role[role] = query_sentence_kb(
            persist_dir=persist_dir,
            query_text=str(text),
            n_results=n_results_each,
            collection_name=collection_name,
            # treat chunk key as sentence_role:
            sentence_role=None,
            # common filters:
            case_id=case_id,
            failure_id=failure_id,
            cause_id=cause_id,
            source_section=role_source_section,
            status=status,
            subject=subject,
            min_faithful_score=faithful_score,
            productPnID=productPnID,
            product_domain=product_domain,
            extra_where=extra_where,
            include=["documents", "metadatas", "distances"],
        )

    # aggregate by group_by
    # agg = defaultdict(lambda: {"group_id": None, "score": 0.0, "hits": []})
    agg = defaultdict(lambda: {"group_id": None, "score": 0.0,"hits": [],
    "best_by_chunk": {}   # sentence_id -> best hit
    })

    for role, r in by_role.items():
        ids0 = r.get("ids", [[]])[0]
        docs0 = r.get("documents", [[]])[0]
        metas0 = r.get("metadatas", [[]])[0]
        dists0 = r.get("distances", [[]])[0]

        for rid, doc, meta, dist in zip(ids0, docs0, metas0, dists0):
            gid = meta.get(group_by)
            if not gid:
                continue

            a = agg[gid]
            a["group_id"] = gid

            w = CHUNK_WEIGHT.get(role, 1.0)
            s = w / (1.0 + float(dist))

            best = a["best_by_chunk"].get(rid)

            # 只保留这个 sentence 的“最佳贡献”
            if best is None or s > best["score"]:
                a["best_by_chunk"][rid] = {
                    "score": s,
                    "sentence_id": rid, 
                    "distance": dist,
                    "chunk": role,
                    "text": doc,
                    "metadata": meta,
                    "from_chunk": role,
                }
        for a in agg.values():
            seen = set()
            total = 0.0
            hits = []

            for info in sorted(a["best_by_chunk"].values(), key=lambda x: x["score"], reverse=True):
                sid = info["sentence_id"]
                if sid in seen:
                    continue
                seen.add(sid)

                hits.append(info)
                total += info["score"]

            a["hits"] = hits
            a["score"] = total
    merged = sorted(agg.values(), key=lambda x: x["score"], reverse=True)
    return {"by_role": by_role, "merged": merged}



if  __name__ == "__main__":
    KB_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\sentence_kb")
    # result = get_sentences_by_ids(persist_dir=KB_PATH, ids=["8D66491000161R01_D4_S004"])
    # result = get_sentences_by_metadata(persist_dir=KB_PATH,limit=5,case_id="8D6649140050R06", source_section="D3", min_faithful_score=90)
    # print(result)
    FAILURE_ENTITY = {
        "failure_mode": "motor stall after prolonged operation",
        "failure_element": "motor drive",
        "failure_effect": "",
        "failure_cause": "controller software timer logic",
    }

    out = query_sentence_kb_by_chunks(
        persist_dir=KB_PATH,
        entity=FAILURE_ENTITY,
        n_results_each=5,
        # source_type=["new_fmea", "old_fmea"],   # optional filter
        # productPnID=287883,                    # 
    )
    for role, res in out["by_role"].items():
        print("\n" + "=" * 80)
        print(f"[ROLE QUERY] {role}")

        ids0   = res.get("ids", [[]])[0]
        docs0  = res.get("documents", [[]])[0]
        metas0 = res.get("metadatas", [[]])[0]
        dists0 = res.get("distances", [[]])[0]

        for i, (rid, doc, meta, dist) in enumerate(
            zip(ids0, docs0, metas0, dists0), start=1
        ):
            print(f"  #{i} id={rid} dist={float(dist):.4f}")
            print(
                f"     case_id={meta.get('case_id')} "
                f"sentence_role={meta.get('sentence_role')} "
                # f"source={meta.get('source_type')} "
                f"subject={meta.get('subject')} "
                f"pn={meta.get('productPnID')}"
            )
            print(f"     text={doc}")

    # ------------------------------------------------------------------
    # 2) Aggregated (merged) result
    # ------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("[MERGED RESULT TOP]")

    for rank, item in enumerate(out["merged"][:10], start=1):
        print(
            f"\n[{rank}] group_id={item['group_id']} "
            f"(group_by=failure_id) score={item['score']:.4f}"
        )

        for h in item["hits"]:
            meta = h["metadata"]
            print(
                f"   - from_chunk={h['from_chunk']} "
                f"dist={float(h['distance']):.4f} "
                f"sentence_id={h['sentence_id']} "
                f"sentence_role={h['metadata'].get('sentence_role')}"
            )
            print(f"     {h['text']}")