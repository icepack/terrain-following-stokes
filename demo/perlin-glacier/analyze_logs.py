import numpy as np

with open("perlin.log", "r") as log_file:
    all_lines = log_file.readlines()

converged_reasons = [
    line for line in all_lines
    if "Nonlinear firedrake_" in line
]

num_steps = np.array([int(text.split()[-1]) for text in converged_reasons])
print(f"Maximum number of Newton iterations: {num_steps.max()}")
