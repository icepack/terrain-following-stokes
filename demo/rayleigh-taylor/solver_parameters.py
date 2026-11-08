
lu = {
    "snes_linesearch_type": "nleqerr",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_7": 5,
    "mat_mumps_icntl_14": 100,
}


newton_krylov_rana = {
    # Outer Newton on the true Radau system
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_max_it": 30,
    "snes_atol": 1e-8,
    "snes_rtol": 0.0,
    "snes_stol": 0.0,
    # Operator: "aij" assembles the coupled Jacobian
    "mat_type": "matfree",
    # Preconditioner matrix: assembled from the Rana LD form
    "pmat_type": "aij",
    # Inexact linear solve
    "ksp_type": "gmres",
    "ksp_rtol": 1e-3,
    "ksp_max_it": 60,
    # Rana LD is block lower triangular, so a multiplicative
    # split by stage is an exact forward substitution
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_0_fields": "0,1,2,3",   # stage 1: u, p, ρ, h
    "pc_fieldsplit_1_fields": "4,5,6,7",   # stage 2: u, p, ρ, h
    "fieldsplit_ksp_type": "preonly",
    "fieldsplit_pc_type": "lu",
    "fieldsplit_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_mat_mumps_icntl_7": 5,
    "fieldsplit_mat_mumps_icntl_14": 100,
}
