from solver import Solver
from plot import plot_rectangles

from typing import *

import numpy as np
import time


def main():
    solver = Solver(37, 56, [1.3 + i * 0.4 for i in range(4)])

    initital_guess = [20, 28, 10, 10, 0.5] * 4

    start = time.perf_counter()
    res = solver.global_optimize(x0=initital_guess)
    end = time.perf_counter()

    print(f"Time taken to calculate: {end - start}, fun: {res['fun']}")

    rectangles = res["solution"]

    plot_rectangles(rectangles)


if __name__ == "__main__":
    main()
