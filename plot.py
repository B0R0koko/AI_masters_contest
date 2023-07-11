import matplotlib.pyplot as plt
import matplotlib.patches as patches

from solver import Rectangle
from typing import *


def plot_rectangles(rectangles: List[Rectangle]):
    fig, ax = plt.subplots(figsize=(16, 10))
    colors = list("bgrcmykw")

    for rect, color in zip(rectangles, colors):
        rectp = patches.Rectangle(
            (rect.x, rect.y), rect.w, rect.h, alpha=0.2, facecolor=color
        )
        ax.add_patch(rectp)

    ax.set_aspect("equal")

    ax.set_ylim(0, 56)
    ax.set_xlim(0, 37)

    plt.show()
