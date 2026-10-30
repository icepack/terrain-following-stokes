import numpy as np
from collections import Counter


def read_log(filename: str) -> np.ndarray:
    with open(filename, "r") as log_file:
        all_lines = log_file.readlines()

    converged_reasons = [
        line for line in all_lines
        if "Nonlinear firedrake_" in line
    ]

    return np.array([int(text.split()[-1]) for text in converged_reasons])


filenames = {
    "euler": "perlin.log",
    "degree=3": "perlin-degree3.log",
    "nz=10": "perlin-nz10.log",
    "radau": "perlin-radau.log",
}

logs = {key: read_log(filename) for key, filename in filenames.items()}

for key, log in logs.items():
    print(f"{key}: {Counter(log)}")
