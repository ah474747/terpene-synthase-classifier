import json

def load_class_to_type(label_order_path="models/checkpoints/label_order.json",
                       class_to_type_path="models/checkpoints/class_to_type.json"):
    labels = json.load(open(label_order_path))
    c2t_names = json.load(open(class_to_type_path))
    # stable type ordering
    types = []
    for c in labels:
        t = c2t_names.get(c, "general")
        if t not in types: types.append(t)
    type_to_id = {t:i for i,t in enumerate(types)}
    class_to_type_id = {}
    for c, tname in c2t_names.items():
        if c in labels:
            class_to_type_id[labels.index(c)] = type_to_id.get(tname, type_to_id["general"])
    return class_to_type_id, type_to_id


