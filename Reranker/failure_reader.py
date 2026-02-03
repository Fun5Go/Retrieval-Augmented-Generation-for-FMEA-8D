import json

ROLES = ["failure_mode", "failure_element", "failure_effect", "failure_cause"]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def complete_out_json_by_cause_id_with_source_type(
    out_json_path: str,
    kb_8d_path: str,
    kb_fmea_new_path: str,
    kb_fmea_old_path: str,
    out_path: str,
):
    out_obj = load_json(out_json_path)  # {"results":[...]}
    kb_8d = load_json(kb_8d_path)       # {cause_id: {...}}
    kb_new = load_json(kb_fmea_new_path)
    kb_old = load_json(kb_fmea_old_path)

    kb_map = {
        "8D": kb_8d,
        "newFMEA": kb_new,
        "oldFMEA": kb_old,
    }

    miss = 0
    bad_src = 0

    for it in out_obj.get("results", []):
        cid = it.get("cause_id")
        src = it.get("source_type")
        if not cid or not src:
            miss += 1
            continue

        kb = kb_map.get(src)
        if kb is None:
            bad_src += 1
            continue

        rec = kb.get(cid)
        if not rec:
            miss += 1
            continue

        # 合并：不覆盖 result 已有键
        for k, v in rec.items():
            it.setdefault(k, v)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(
        f"saved={out_path}, results={len(out_obj.get('results', []))}, "
        f"kb_miss={miss}, bad_source_type={bad_src}"
    )
    return out_obj


# example:
# complete_out_json_by_cause_id_with_source_type(
#     out_json_path="out.json",
#     kb_8d_path="8d.json",
#     kb_fmea_new_path="fmea_new.json",
#     kb_fmea_old_path="fmea_old.json",
#     out_path="out.completed.json",
# )