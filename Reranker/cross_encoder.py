from sentence_transformers import CrossEncoder
import json

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2")

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
            return failure

    elif source_type in {"new_fmea", "old_fmea"}:
        if cid and cid in fmea_json:
            failure = fmea_json[cid]
            return failure

    return None

# ======== 8D sentence ========
def _join_topk_sentences(sentences_by_role: dict, role: str, k: int = 3) -> str:
    """sentences_by_role[role] = [{text, distance, ...}, ...]"""
    sents = (sentences_by_role or {}).get(role) or []
    top = sents[:k]
    return "\n".join([t.get("text", "") for t in top if t.get("text")])
def _ce_score(query_text: str, text: str):
    if not text or not text.strip():
        return None
    return float(cross_encoder.predict([(query_text, text)])[0])



def rerank_failures(
    query_entity: dict,
    retriever_results: list,
    fmea_json: dict,
    fmea_8d_json: dict,
    top_k: int = 5,
    use_doc_score_for_8d: bool = True,   
    alpha_doc: float = 0.3,    
):
    if isinstance(retriever_results, dict):
        if "results" in retriever_results:
            retriever_results = retriever_results["results"]
        else:
            raise ValueError("retriever_results dict has no 'results' key")
    role_weights_8dsentence = {
        "failure_element": 0.2,
        "failure_mode": 0.3,
        "failure_effect": 0.2,
        "failure_cause": 0.3,
    }

    before_rank_map = {}
    for idx, hit in enumerate(retriever_results, start=1):
        cid = hit.get("cause_id")  
        if cid and cid not in before_rank_map:
            before_rank_map[cid] = idx

    query_text = build_query_text(query_entity)

    scored = []

    for hit in retriever_results:
        failure = load_full_failure(hit, fmea_json, fmea_8d_json)
        if not failure:
            continue

        source_type = failure.get("source_type") or hit.get("source_type")
        cause_id = failure.get("cause_id")

        candidate_text = build_failure_text(failure)

        failure_score = _ce_score(query_text, candidate_text) or float("-inf")

        final_score = failure_score
        debug = {"doc_score": failure_score}

        #====== Sentence crosss encoder==== (not good)
        # if source_type == "8D":
        #     sentences_by_role = hit.get("sentences_by_role") or {}
        #     role_scores = {}

        #     for role, w in role_weights_8dsentence.items():
        #         role_text = _join_topk_sentences(sentences_by_role, role, k=3)
        #         s = _ce_score(query_text, role_text)
        #         if s is not None:
        #             role_scores[role] = s

        #     debug["role_scores"] = role_scores

        #     if role_scores:
        #         w_sum = sum(role_weights_8dsentence[r] for r in role_scores.keys())
        #         sent_agg = sum(role_weights_8dsentence[r] * role_scores[r] for r in role_scores.keys()) / (w_sum or 1.0)
        #     else:
        #         sent_agg = float("-inf")

        #     debug["sent_agg"] = sent_agg

        #     # 
        #     if use_doc_score_for_8d and failure_score != float("-inf") and sent_agg != float("-inf"):
        #         final_score = alpha_doc * failure_score + (1 - alpha_doc) * sent_agg
        #     else:
        #         final_score = sent_agg if sent_agg != float("-inf") else failure_score

        #     debug["final"] = final_score

        cause_id = failure.get("cause_id") 
        scored.append({
                    "failure": failure,
                    "source_type": source_type,
                    "rerank_score": float(final_score),
                    "before_rank": before_rank_map.get(cause_id),
                    "debug": debug,
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

        br = r.get("before_rank")

        print("=" * 90)
        print(f"Before rank: {br}")
        print(f"Rank:        {i}")
        print(f"Score:       {r.get('rerank_score')}")
        print(f"Source:      {r.get('source_type')}")
        print(f"Failure ID:  {f.get('failure_id')}")
        print(f"Cause ID:    {f.get('cause_id')}")
        print("-" * 90)
        print(f"Failure mode:    {failure_mode}")
        print(f"Element:         {failure_element}")
        print(f"Effect:          {failure_effect}")
        print(f"Cause:           {failure_cause}")
        dbg = r.get("debug") or {}
        if dbg:
            print("-" * 90)
            print("Debug:")
            print(json.dumps(dbg, indent=2, ensure_ascii=False))

        if show_failure_json:
            print("-" * 90)
            print("Full failure JSON:")
            print(json.dumps(f, indent=2, ensure_ascii=False))

    print("=" * 90)


if __name__ == "__main__":

    FAILURE_ENTITY = {
    "failure_mode": "Current trip during run",
    "failure_element": "Motor drive",
    "failure_effect": "",
    "failure_cause": "Insufficient margin between SW limit and HW trip"
        }
    ED_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\8d_cause_store.json"
    FMEA_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\fmea_cause_store.json"
    RETRIEVER_RESULT = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\out.json"

    ed_store = load_json(ED_JSON)          # 8D store dict
    fmea_store = load_json(FMEA_JSON)      # fmea store dict
    out = load_json(RETRIEVER_RESULT)      # out.json dict
    retriever_results = out["results"] if isinstance(out, dict) else out
    # print(retriever_results)

    reranked = rerank_failures(query_entity=FAILURE_ENTITY, retriever_results=retriever_results,fmea_8d_json=ed_store, fmea_json=fmea_store, top_k=10)
    print_reranked(reranked, top_n=10, show_failure_json=False)


