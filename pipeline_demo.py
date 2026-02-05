from Retriever.integration_tools import query_fmea_top_failures_with_sentences
from Reranker.cross_encoder import rerank_failures,print_reranked
# from Reranker.Regroup import  regroup_function
import json
from pathlib import Path 

def save_json(out: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    FAILURE_ENTITY = {
    "failure_mode": "Current trip during run",
    "failure_element": "Motor drive",
    "failure_effect": "",
    "root_cause": "Insufficient margin between SW limit and HW trip"
        }
    ED_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\8d_cause_store.json"
    FMEA_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\fmea_cause_store.json"
    RETRIEVER_RESULT = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\out.json"
    SENTENCE_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\sentence_kb")
    FAILURE_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb")

    retriever_result = query_fmea_top_failures_with_sentences(
    failure_kb_dir=FAILURE_PATH,
    sentence_kb_dir=SENTENCE_PATH,
    entity=FAILURE_ENTITY,
    top_n_failures = 30,
    n_results_each=12,
    sentence_top_k=3,
    product_domain="motor_drives",
    # source_type="new_fmea",
    )
    save_json(retriever_result, "retrieval_list.json")

    ed_store = load_json(ED_JSON)          # 8D store dict
    fmea_store = load_json(FMEA_JSON)      # fmea store dict
    retriever_results = retriever_result["results"] if isinstance(retriever_result, dict) else retriever_result
    # print(retriever_results)


    reranked_result = rerank_failures(query_entity=FAILURE_ENTITY, retriever_results=retriever_results,fmea_8d_json=ed_store, fmea_json=fmea_store, top_k=10)
    save_json(reranked_result, "reranked_list.json")
    print_reranked(reranked_result, top_n=5, show_failure_json=False)

