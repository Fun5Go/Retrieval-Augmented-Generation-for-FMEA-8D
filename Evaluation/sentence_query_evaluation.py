import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Retriever.sentence_query_tools import query_sentence_kb_by_chunks, query_sentence_kb_by_concat


GT_JSON_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\8d_sample_10pct_rephrased.json"
PERSIST_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\sentence_kb"  
TOP_K = 5
N_RESULTS_EACH_ROLE = 12


def get_predicted_failure_ids(result: Dict[str, Any], k: int = TOP_K) -> List[str]:
    """
    Extract top-k predicted failure_ids from query_sentence_kb_by_chunks output.
    Assumes result["merged"] items contain {"group_id": <failure_id>} when group_by="failure_id".
    """
    merged = result.get("merged", []) or []
    pred = []
    for row in merged:
        fid = row.get("group_id")  # group_id == failure_id (when group_by="failure_id")
        if fid:
            pred.append(str(fid))
        if len(pred) >= k:
            break
    return pred


def reciprocal_rank(pred: List[str], gt: str) -> float:
    for i, p in enumerate(pred, start=1):
        if p == gt:
            return 1.0 / i
    return 0.0


def recall_at_k(pred: List[str], gt: str, k: int) -> float:
    return 1.0 if gt in pred[:k] else 0.0


def build_entity(item: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Map json item -> chunks input expected by query_sentence_kb_by_chunks.
    query_sentence_kb_by_chunks :
      failure_mode / failure_element / failure_effect / failure_cause
    """
    return {
        "failure_mode": item.get("failure_mode"),
        "failure_element": item.get("failure_element"),
        "failure_effect": item.get("failure_effect"),
        "failure_cause": item.get("root_cause"),
    }


def get_predicted_groups_with_sentences(result: Dict[str, Any], k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Return top-k predicted groups with their sentences(hits).
    Each group: {failure_id, score, sentences:[...]}
    """
    merged = result.get("merged", []) or []
    out = []
    for row in merged[:k]:
        fid = row.get("group_id")
        if not fid:
            continue
        hits = row.get("hits", []) or []
        out.append({
            "failure_id": str(fid),
            "score": row.get("score", 0.0),
            # 只保留必要字段；想全量也可以直接返回 hits
            # "sentences": [
            #     {
            #         "sentence_id": h.get("sentence_id"),
            #         "chunk": h.get("chunk") or h.get("from_chunk"),
            #         "distance": h.get("distance"),
            #         "score": h.get("score"),
            #         "text": h.get("text"),
            #         "metadata": h.get("metadata"),
            #     }
            #     for h in hits
            # ],
        })
    return out

def get_predicted_failure_ids_from_concat(result, k):
    merged = result.get("merged", []) or []
    out = []

    for row in merged:
        # 从 hits -> metadata 里读 failure_id
        hits = row.get("hits") or []
        fid = None
        for h in hits:
            meta = h.get("metadata") or {}
            fid = meta.get("failure_id")
            if fid:
                break

        if fid:
            out.append(str(fid))

        if len(out) >= k:
            break

    return out


def get_predicted_failure_ids_from_groups(pred_groups: List[Dict[str, Any]], k: int = TOP_K) -> List[str]:
    return [g["failure_id"] for g in pred_groups[:k]]

def evaluate_sentence(
    gt_data: Dict[str, Dict[str, Any]],
    persist_dir: str,
    top_k: int = TOP_K,
    n_results_each_role: int = N_RESULTS_EACH_ROLE,
    # 可选过滤器（按你的 sentence KB metadata 来）
    case_id: Optional[str] = None,
    cause_id: Optional[str] = None,
    source_section: Optional[str] = None,
    status: Optional[str] = None,
    subject: Optional[str] = None,
    faithful_score: Optional[int] = None,
    productPnID: Optional[str] = None,
    product_domain: Optional[str] = None,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    precisions, recalls, details = [], [], []

    for key, item in gt_data.items():
        # 这里用 failure_id 作为 GT
        gt_failure_id = item.get("failure_id")
        if not gt_failure_id:
            # 没有 failure_id 的样本跳过（或你也可以 raise）
            continue
        gt_failure_id = str(gt_failure_id)

        entity = build_entity(item)

        result = query_sentence_kb_by_chunks(
            persist_dir=persist_dir,
            entity=entity,
            n_results_each=n_results_each_role,
            collection_name="sentences",
            # 这里不要传 failure_id 过滤，否则等于把答案“喂给检索器”
            failure_id=None,
            case_id=case_id,
            cause_id=cause_id,
            source_section=source_section,
            status=status,
            subject=subject,
            faithful_score=faithful_score,
            productPnID=productPnID,
            product_domain=product_domain,
            group_by="failure_id",
        )
        #========== Failure sentence query
        # result = query_sentence_kb_by_concat(
        # persist_dir=persist_dir,
        # entity=entity,
        # top_k=60,
        # )

        # pred_topk = get_predicted_failure_ids(result, k=top_k)

        pred_groups_topk = get_predicted_groups_with_sentences(result, k=top_k)
        pred_topk = get_predicted_failure_ids_from_groups(pred_groups_topk, k=top_k)

        # pred_topk = get_predicted_failure_ids_from_concat(result, k=TOP_K)

        p = reciprocal_rank(pred_topk, gt_failure_id)
        r = recall_at_k(pred_topk, gt_failure_id, top_k)

        precisions.append(p)
        recalls.append(r)

        details.append({
            "item_key": key,
            "gt_failure_id": gt_failure_id,
            "pred_topk_failure_id": pred_topk,
            "pred_groups_topk": pred_groups_topk,   # 这里包含对应句子
            "hit_at_k": (gt_failure_id in pred_topk),
            "precision_at_k": p,
            "recall_at_k": r,
            "entity": entity,
        })

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0
    return avg_p, avg_r, details


def main():
    with open(GT_JSON_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    avg_p, avg_r, details = evaluate_sentence(
        gt_data=gt_data,
        persist_dir=PERSIST_DIR,
        top_k=TOP_K,
        n_results_each_role=N_RESULTS_EACH_ROLE,
        # product_domain="motor drive",
        # productPnID="xxx",
    )

    print(f"Evaluated items: {len(details)}")
    print(f"Precision@{TOP_K}: {avg_p:.4f}")
    print(f"Recall@{TOP_K}:    {avg_r:.4f}")

    out_path = Path(f"eval_sentence_failureid_top{TOP_K}_details.json")
    out_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved details to: {out_path.resolve()}")


if __name__ == "__main__":
    main()