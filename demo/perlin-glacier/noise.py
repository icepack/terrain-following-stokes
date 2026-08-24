import numpy as np
from scipy.interpolate import CubicHermiteSpline


def generate_octave(num_intervals, rng):
    xs = np.linspace(0.0, 1.0, num_intervals + 1)
    ys = np.zeros_like(xs)
    dy_dx = rng.uniform(-1, +1, len(xs))
    dy_dx[0] = 0.0
    dy_dx[-1] = 0.0
    return CubicHermiteSpline(xs, ys, dy_dx)


class PerlinNoise:
    def __init__(self, amplitudes, rng):
        self.amplitudes = amplitudes
        num_octaves = len(amplitudes)
        self.octaves = [
            generate_octave(2 ** index, rng) for index in range(num_octaves)
        ]

    def __call__(self, xs):
        modes = np.array([octave(xs) for octave in self.octaves])
        return self.amplitudes @ modes
