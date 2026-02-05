import json
from collections import defaultdict

def _group_from_source_type(source_type: str) -> str:
    """Map source_type to a coarse group."""
    st = (source_type or "").strip().lower()
    if st in {"new_fmea", "old_fmea"}:
        return "FMEA"
    if st == "8d":
        return "8D"
    return "OTHER"

def _year_bonus(released_year) -> float:
    """Year bonus: year > 2018 => +0.5, else +0.0."""
    try:
        y = int(released_year)
    except Exception:
        return 0.0
    return 0.5 if y > 2018 else 0.0

def _confidence_value(conf) -> float:
    """
    8D confidence value:
      high   => +0.3
      medium => +0.0
      low    => -0.3
    """
    c = (conf or "").strip().lower()
    if c == "high":
        return 0.3
    if c == "medium" or c == "":
        return 0.0
    if c == "low":
        return -0.3
    return 0.0

def regroup_function(results: list):
    """
    Keep original retriever score unchanged.
    Create a new field '_value' and '_final_score' = score + _value for later use.

    Output only two groups: FMEA and 8D (OTHER is ignored).
    """
    groups = {"FMEA": [], "8D": []}

    for i, item in enumerate(results):
        it = dict(item)
        st = (it.get("source_type") or "").strip().lower()
        group = _group_from_source_type(st)

        if group not in groups:
            continue

        # Preserve original rank (1-based) from retriever output order
        it["_orig_rank"] = i + 1
        it["_group"] = group

        # Initialize value
        value = 0.0

        # Type-based value
        if st == "new_fmea":
            value += 0.4
        elif st == "old_fmea":
            value += 0.0
        elif st == "8d":
            value += _confidence_value(it.get("confidence"))

        # Year-based value (same rule for FMEA and 8D)
        value += _year_bonus(it.get("released_year"))

        it["_value"] = float(value)
        it["_final_score"] = float(it.get("score", 0.0)) + it["_value"]

        groups[group].append(it)

    return {"groups": groups}


if __name__ == "__main__":
    RETRIEVER_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\out.json"

    with open(RETRIEVER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    post = regroup_function(data["results"])

    # Optional: save
    OUT_PATH = RETRIEVER_PATH.replace(".json", ".grouped.json")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(post, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {OUT_PATH}")
    print(f"FMEA items: {len(post['groups']['FMEA'])}, 8D items: {len(post['groups']['8D'])}")