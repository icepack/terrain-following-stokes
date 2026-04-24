from firedrake import ds_b, ds_t, ds_v


# TODO: make this less repulsive
def boundary_measure(ids):
    numeric_ids = tuple(set(ids) - {"bottom", "top"})
    measures = []
    if numeric_ids:
        measures.append(ds_v(numeric_ids))
    if "bottom" in ids:
        measures.append(ds_b)
    if "top" in ids:
        measures.append(ds_t)
    result = measures.pop()
    while measures:
        result += measures.pop()
    return result

