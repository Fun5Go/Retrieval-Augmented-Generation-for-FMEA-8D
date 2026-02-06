from collections import defaultdict
from integration_tools import query_fmea_top_failures_with_sentences

def expand_structure_to_queries(structure_nodes, max_pairs_per_element=30):
    queries = []
    for node in structure_nodes:
        element = node.get("failure_element", "")
        modes = node.get("modes", []) or [""]
        causes = node.get("causes", []) or [""]
        effects = node.get("effects", []) or [""]

        pairs = []
        for m in modes:
            for c in causes:
                pairs.append((m, c))
        pairs = pairs[:max_pairs_per_element]

        for (m, c) in pairs:
            for eff in effects[:1]:
                queries.append({
                    "element_id": node.get("element_id"),
                    "entity": {
                        "failure_mode": m,
                        "failure_element": element,
                        "failure_effect": eff,
                        "root_cause": c
                    }
                })
    return queries


def batch_retrieve_rerank(structure_input, failure_kb_dir, sentence_kb_dir,
                         ed_store, fmea_store,
                         top_n_failures=30, n_results_each=12, sentence_top_k=3, top_k=10):

    queries = expand_structure_to_queries(structure_input["nodes"])

    per_query_results = []
    agg = defaultdict(list)  # failure_id -> list of supports

    for q in queries:
        entity = q["entity"]
        retriever_result = query_fmea_top_failures_with_sentences(
            failure_kb_dir=failure_kb_dir,
            sentence_kb_dir=sentence_kb_dir,
            entity=entity,
            top_n_failures=top_n_failures,
            n_results_each=n_results_each,
            sentence_top_k=sentence_top_k,
            product_domain=structure_input.get("product_domain")
        )
        retriever_results = retriever_result["results"] if isinstance(retriever_result, dict) else retriever_result

        reranked = rerank_failures(
            query_entity=entity,
            retriever_results=retriever_results,
            fmea_8d_json=ed_store,
            fmea_json=fmea_store,
            top_k=top_k
        )

        # 记录每个query的top命中
        per_query_results.append({
            "element_id": q["element_id"],
            "query": entity,
            "matches": reranked
        })

        # 聚合：假设 reranked item里有 failure_id 和 score
        for item in reranked:
            fid = item.get("failure_id") or item.get("id")  # 按你数据结构调整
            if not fid:
                continue
            agg[fid].append({
                "element_id": q["element_id"],
                "mode": entity.get("failure_mode", ""),
                "cause": entity.get("root_cause", ""),
                "score": item.get("score", 0.0)
            })

    # 全局去重排序：可以用 max/mean/加权
    global_ranked = []
    for fid, supports in agg.items():
        global_score = max(s["score"] for s in supports)  # 简单可用
        global_ranked.append({"failure_id": fid, "global_score": global_score, "supported_by": supports})
    global_ranked.sort(key=lambda x: x["global_score"], reverse=True)

    return {"per_query": per_query_results, "global": global_ranked}