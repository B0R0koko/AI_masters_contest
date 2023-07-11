import numpy as np

from scipy.optimize import minimize, basinhopping
from math import floor

from itertools import combinations
from typing import *


class Rectangle:
    # x, y coordinates of the Rectangle as well as its width and height and rotation boolean
    def __init__(self, x: float, y: float, w: float, h: float, g: float):
        self.x = x
        self.y = y
        self.w = h if g else w
        self.h = w if g else h

    def __str__(self):
        return f"Rectangle: bottom right corner: ({self.x}, {self.y}); width: {self.w}, height: {self.h}"

    def __repr__(self):
        return f"Rectangle: ({self.x}, {self.y}), ({self.w}, {self.h})"


class Solver:
    base_vars = ["x", "y", "w", "h", "g"]

    def __init__(self, W: int, H: int, ratios: List[float]) -> Self:
        self.W = W
        self.H = H
        self.ratios = ratios
        self.n_boxes = len(ratios)
        self.vars = self.generate_vars()

        eps = 1e-4

        self.bounds = [
            (eps, self.W),
            (eps, self.H),
            (eps, self.W),
            (eps, self.H),
            (-eps, 1 + eps),
        ] * self.n_boxes

    def generate_vars(self) -> List[str]:
        return [f"{var}{i}" for i in range(self.n_boxes) for var in self.base_vars]

    def get_var(self, X: List[str], name: str) -> float:
        return X[self.vars.index(name)]

    def objective_func(self, X: List[float]) -> float:
        # sum of w_i*h_i / (W*H)
        obj = sum(
            [
                self.get_var(X, f"w{i}") * self.get_var(X, f"h{i}")
                for i in range(self.n_boxes)
            ]
        )
        return -obj / (self.W * self.H)

    def constr_ineq(self, X: List[float]) -> np.array:
        constrs = []
        #  # final rectangles are within the boundaries of the box of W and H
        for i in range(self.n_boxes):
            x, y, w, h, g = [
                self.get_var(X, f"{name}{i}") for name in ["x", "y", "w", "h", "g"]
            ]
            # rotated width and height
            wg, wh = w * (1 - g) + h * g, w * g + h * (
                1 - g
            )  # basically just swap the width and height
            constrs.extend(
                [-(x + wg - self.W), -(y + wh - self.H)]  # x + w <= W  # y + h <= H
            )
        # rectangles should not intersect between each other
        for combination in combinations(range(self.n_boxes), 2):
            x1, y1, w1, h1, g1, x2, y2, w2, h2, g2 = [
                self.get_var(X, f"{name}{i}")
                for i in combination
                for name in self.base_vars
            ]
            # swap width and height for each rectangle if it is rotated
            wg1, wg2 = (
                w1 * (1 - g1) + h1 * g1,
                w2 * (1 - g2) + h2 * g2,
            )
            hg1, hg2 = w1 * g1 + h1 * (1 - g1), w2 * g2 + h2 * (1 - g2)

            constrs.append(
                max(
                    [x1 - (x2 + wg2), x2 - (x1 + wg1), y1 - (y2 + hg2), y2 - (y1 + hg1)]
                )
            )

        # add constraints for ratio so that they lie in range -0.1, +0.1
        for i in range(self.n_boxes):
            w, h = [self.get_var(X, f"{name}{i}") for name in ["w", "h"]]
            constrs.extend(
                [
                    -(w / h - self.ratios[i] * 1.1),  # w / h <= 1.1 ratio
                    w / h - self.ratios[i] * 0.9,
                ]
            )

        return np.array(constrs)

    def constr_eq(self, X: List[float]) -> np.array:
        constrs = []

        for i in range(self.n_boxes):
            x, y, w, h, g = [
                self.get_var(X, f"{name}{i}") for name in ["x", "y", "w", "h", "g"]
            ]
            constrs.extend(
                [
                    g * (1 - g),  # add a constraint that enforce gi being either 1 or 0
                    # x - int(x),
                    # y - int(y),
                    # w - int(w),
                    # h - int(h),
                ]
            )

        return np.array(constrs)

    # this will yield local extrema of the function
    def optimize(self, x0: List[float]) -> Dict[str, Any]:
        cons = [
            {"type": "ineq", "fun": self.constr_ineq},
            {"type": "eq", "fun": self.constr_eq},
        ]
        res = minimize(
            fun=self.objective_func,
            x0=x0,
            bounds=self.bounds,
            constraints=cons,
            method="SLSQP",
        )
        return {
            "message": res.message,
            "status": res.status,
            "fun": -res.fun,
            "solution": [
                Rectangle(*res.x[i * 5 : (i + 1) * 5].round(3))
                for i in range(self.n_boxes)
            ],
        }

    def global_optimize(self, x0: List[float]) -> Dict[str, Any]:
        cons = [
            {"type": "ineq", "fun": self.constr_ineq},
            {"type": "eq", "fun": self.constr_eq},
        ]
        res = basinhopping(
            func=self.objective_func,
            x0=x0,
            niter=200,
            minimizer_kwargs={
                "method": "SLSQP",
                "bounds": self.bounds,
                "constraints": cons,
            },
        )
        return {
            "message": res.message,
            "fun": -res.fun,
            "solution": [
                Rectangle(*res.x[i * 5 : (i + 1) * 5].round(3))
                for i in range(self.n_boxes)
            ],
        }


if __name__ == "__main__":
    solver = Solver(10, 10, [0.5, 2, 3])
    res = solver.global_optimize(x0=[0.5] * 15)
    print(res)
