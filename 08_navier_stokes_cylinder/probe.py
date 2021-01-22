from argparse import ArgumentParser
from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np

parser = ArgumentParser()
parser.add_argument("history", type=Path)
args = parser.parse_args()

time, fore, aft = np.loadtxt(args.history, delimiter=",").T

fig, axs = subplots(2, 2, sharex="col", sharey="row")
axs[1][0].plot(fore, aft, marker=",", linestyle="None")
axs[1][0].set_xlabel("fore")
axs[1][0].set_ylabel("aft")
axs[0][0].plot(fore, time, marker=",", linestyle="None")
axs[0][0].set_ylabel("time")
axs[1][1].plot(time, aft, marker=",", linestyle="None")
axs[1][1].set_xlabel("time")
axs[0][1].axis("off")
axs[1][0].set_xlim(1.9, 2.1)
axs[1][0].set_ylim(-0.5, -0.3)
fig.suptitle("Pressure at fore and aft stagnation points")
fig.savefig(Path(__file__).with_suffix(".png"))
