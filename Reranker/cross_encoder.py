from sentence_transformers import CrossEncoder
import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_query_text(entity: dict) -> str:
    return (
        f"Failure element: {entity.get('failure_element', '')}. "
        f"Failure mode: {entity.get('failure_mode', '')}. "
        f"Failure effect: {entity.get('failure_effect', '')}. "
        f"Failure cause: {entity.get('failure_cause', '')}."
    )

def build_failure_text(failure: dict) -> str:
    return (
        f"Failure element: {failure.get('failure_element', '')}. "
        f"Failure mode: {failure.get('failure_mode', '')}. "
        f"Failure effect: {failure.get('failure_effect', '')}. "
        f"Failure cause: {failure.get('failure_cause', '')}."
    )

def load_full_failure(hit, fmea_json, fmea_8d_json):
    source_type = hit["source_type"]
    cid = hit.get("cause_id")

    if source_type == "8D":
        if cid and cid in fmea_8d_json:
            failure = fmea_8d_json[cid]
            failure["_cause_id"] = cid
            failure["_failure_id"] = hit.get("failure_id")
            return failure

    elif source_type in {"new_fmea", "old_fmea"}:
        if cid and cid in fmea_json:
            failure = fmea_json[cid]
            failure["_cause_id"] = cid
            failure["_failure_id"] = hit.get("failure_id")
            return failure

    return None


cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
def rerank_failures(
    query_entity: dict,
    retriever_results: list,
    fmea_json: dict,
    fmea_8d_json: dict,
    top_k: int = 5,
):
    if isinstance(retriever_results, dict):
        if "results" in retriever_results:
            retriever_results = retriever_results["results"]
        else:
            raise ValueError("retriever_results dict has no 'results' key")
    query_text = build_query_text(query_entity)

    scored = []

    for hit in retriever_results:
        failure = load_full_failure(hit, fmea_json, fmea_8d_json)
        if not failure:
            continue

        candidate_text = build_failure_text(failure)

        score = cross_encoder.predict(
            [(query_text, candidate_text)]
        )[0]

        scored.append({
        "failure_id": failure.get("_failure_id"),
        "cause_id": failure.get("_cause_id"),
                    "source_type": hit.get("source_type"),
                    "rerank_score": float(score),
                    "failure": failure,
                })

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]

def _pick(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] not in (None, "", [], {}):
            return d[k]
    return default

def print_reranked(reranked, top_n=3, show_failure_json=False):
    for i, r in enumerate(reranked[:top_n], start=1):
        f = r.get("failure", {}) or {}

        # try a few common key names; adjust to your schema if needed
        failure_mode   = _pick(f, ["failure_mode", "Failure Mode", "mode"])
        failure_element= _pick(f, ["failure_element", "Element", "item"])
        failure_effect = _pick(f, ["failure_effect", "Effect"])
        failure_cause  = _pick(f, ["failure_cause", "Cause"])

        print("=" * 90)
        print(f"Rank:        {i}")
        print(f"Score:       {r.get('rerank_score'):.4f}")
        print(f"Source:      {r.get('source_type')}")
        print(f"Failure ID:  {r.get('failure_id')}")
        print(f"Cause ID:    {r.get('cause_id')}")
        print("-" * 90)
        print(f"Failure mode:    {failure_mode}")
        print(f"Element:         {failure_element}")
        print(f"Effect:          {failure_effect}")
        print(f"Cause:           {failure_cause}")

        if show_failure_json:
            print("-" * 90)
            print("Full failure JSON:")
            print(json.dumps(f, indent=2, ensure_ascii=False))

    print("=" * 90)


if __name__ == "__main__":

    FAILURE_ENTITY = {
    "failure_mode": "Capacitor short failure",
    "failure_element": "Power electronics",
    "failure_effect": "Motor control lost",
    "failure_cause": "Capacitor mechanically stressed"
        }
    ED_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\8d_cause_store.json"
    FMEA_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\fmea_cause_store.json"
    RETRIEVER_RESULT = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\out.json"

    ed_store = load_json(ED_JSON)          # 8D store dict
    fmea_store = load_json(FMEA_JSON)      # fmea store dict
    out = load_json(RETRIEVER_RESULT)      # out.json dict
    retriever_results = out["results"] if isinstance(out, dict) else out
    # print(retriever_results)


    reranked = rerank_failures(query_entity=FAILURE_ENTITY, retriever_results=retriever_results,fmea_8d_json=ed_store, fmea_json=fmea_store)
    
    print_reranked(reranked, top_n=3, show_failure_json=False)


