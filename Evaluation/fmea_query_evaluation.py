import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # 指向 RAG 根目录
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Retriever.failure_query_tools import query_failure_kb_by_chunks

GT_JSON_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\fmea_sample_10pct_rephrased.json"
PERSIST_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb"  
TOP_K = 5
N_RESULTS_EACH_ROLE = 10  # role-level retrieval size (can tune)

def get_predicted_cause_ids(result: Dict[str, Any], k: int = TOP_K) -> List[str]:
    """Extract top-k predicted cause_ids from query_failure_kb_by_chunks output."""
    merged = result.get("merged", []) or []
    pred = []
    for row in merged:
        cid = row.get("cause_id")
        if cid:
            pred.append(cid)
        if len(pred) >= k:
            break
    return pred

def precision_at_k(pred: List[str], gt: str, k: int) -> float:
    return (1.0 if gt in pred[:k] else 0.0) / float(k)

def reciprocal_rank(pred: List[str], gt: str) -> float:
    for i, p in enumerate(pred, start=1):
        if p == gt:
            return 1.0 / i
    return 0.0

def recall_at_k(pred: List[str], gt: str, k: int) -> float:
    # Single ground-truth per query
    return 1.0 if gt in pred[:k] else 0.0

def build_entity(item: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Map json item to chunks input expected by query_failure_kb_by_chunks."""
    return {
        "failure_mode": item.get("failure_mode"),
        "failure_element": item.get("failure_element"),
        "failure_effect": item.get("failure_effect"),
        "failure_cause": item.get("failure_cause"),
    }

def evaluate(
    gt_data: Dict[str, Dict[str, Any]],
    persist_dir: str,
    top_k: int = TOP_K,
    n_results_each_role: int = N_RESULTS_EACH_ROLE,
    # Optional meta filters if you want to restrict search space:
    source_type: Optional[str] = None,
    productPnID: Optional[str] = None,
    product_domain: Optional[str] = None,
    fmea_type: Optional[str] = None,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    precisions = []
    recalls = []
    details = []

    for key, item in gt_data.items():
        gt_cause_id = item["cause_id"]

        entity = build_entity(item)

        result = query_failure_kb_by_chunks(
            persist_dir=persist_dir,
            entity=entity,
            n_results_each=n_results_each_role,
            source_type=source_type,
            productPnID=productPnID,
            product_domain=product_domain,
            fmea_type=fmea_type,
            #include=["documents", "metadatas", "distances"],  # your function already sets include; safe if ignored
        )

        pred_topk = get_predicted_cause_ids(result, k=top_k)

        p = reciprocal_rank(pred_topk, gt_cause_id)
        r = recall_at_k(pred_topk, gt_cause_id, top_k)

        precisions.append(p)
        recalls.append(r)

        details.append({
            "item_key": key,
            "gt_cause_id": gt_cause_id,
            "pred_topk": pred_topk,
            "hit_at_k": (gt_cause_id in pred_topk),
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

    avg_p, avg_r, details = evaluate(
        gt_data=gt_data,
        persist_dir=PERSIST_DIR,
        top_k=TOP_K,
        n_results_each_role=N_RESULTS_EACH_ROLE,
        # If you want filters, set them here; otherwise leave None:
        # source_type="new_fmea",
        # product_domain="motor drive",
        # fmea_type="system",
    )

    print(f"Evaluated items: {len(details)}")
    print(f"Precision@{TOP_K}: {avg_p:.4f}")
    print(f"Recall@{TOP_K}:    {avg_r:.4f}")

    out_path = Path(f"eval_top{TOP_K}_details.json")
    out_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved details to: {out_path.resolve()}")

if __name__ == "__main__":
    main()