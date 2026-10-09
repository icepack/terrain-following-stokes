import firedrake
from firedrake import ds_b, ds_t, ds_v
from ufl.algorithms import extract_coefficients


def get_test_function(u):
    z, = extract_coefficients(u)
    Z = z.function_space()
    w = firedrake.TestFunction(Z)
    return firedrake.replace(u, {z: w})


def horizontal_ds(boundary_ids):
    subdomain_id_t = "everywhere" if "top" in boundary_ids else ()
    subdomain_id_b = "everywhere" if "bottom" in boundary_ids else ()
    return ds_t(subdomain_id=subdomain_id_t) + ds_b(subdomain_id=subdomain_id_b)


def vertical_ds(boundary_ids):
    numeric_ids = tuple(idx for idx in boundary_ids if isinstance(idx, int))
    return ds_v(subdomain_id=numeric_ids)
