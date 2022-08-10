from argparse import ArgumentParser
from random import SystemRandom
from time import perf_counter_ns

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import Axes, Figure, Text
from numpy import array, log2, ndarray, round


class Game:
    def __init__(
        self,
        size: int = 128,
        threshold: float = 0.93,
        fps: float = 23.976,
        initial_universe: ndarray = None,
        birth_rule: list[int] = None,
        survive_rule: list[int] = None,
        visualize: bool = True,
    ):
        random = SystemRandom()
        plt.style.use("seaborn")
        plt.rcParams.update(
            {
                "figure.figsize": (5, 5),
                "figure.dpi": 200,
            }
        )

        if log2(size) % 1 != 0:
            raise ValueError(f"`size` must be an integer power of 2 (got {size})")
        if birth_rule is None:
            birth_rule = [3]
        if survive_rule is None:
            survive_rule = [2, 3]
        if initial_universe is None:
            initial_universe = array([random.random() > threshold for _ in range(size**2)]).reshape((size, size))

        self.generation = 1
        self.N = size
        self.fps = fps
        self.visualize = visualize
        self.birth_rule = birth_rule
        self.survive_rule = survive_rule
        self.universe: ndarray = initial_universe.astype(bool)
        if self.visualize:
            self.fig: Figure = plt.figure()
            self.ax: Axes = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.N - 1)
            self.ax.set_ylim(0, self.N - 1)
            self.ax.grid(False)
            self.ax.axis("off")
            self.ax.set_title(f"B{''.join([str(b) for b in birth_rule])}/S{''.join([str(s) for s in survive_rule])}")
            self.fig.tight_layout()
            self.img = self.ax.imshow(self.universe, cmap="gray", origin="lower", animated=True)
            self.text_template = "\n".join(
                [
                    "gen: {0}",
                    "req ms: {1:.2f} ({2} fps)",
                    "act ms: {3:.2f} ({4} fps)",
                ]
            )
            self.text: Text = self.ax.text(0.025, 0.025, "", transform=self.ax.transAxes, fontsize=10, color="c")
        self.last_time = perf_counter_ns()

    def init(self):
        self.text.set_text("")
        return self.img, self.text

    def update(self, _):
        N = self.N
        previous_universe = self.universe
        universe = self.universe.copy()
        for j in range(N):
            pure_north = (j + 1) % N
            pure_south = (j - 1) % N
            for i in range(N):
                pure_east = (i + 1) % N
                pure_west = (i - 1) % N
                neighbors = [
                    (pure_north, i),  # N
                    (pure_south, i),  # S
                    (j, pure_east),  # E
                    (j, pure_west),  # W
                    (pure_north, pure_west),  # NW
                    (pure_north, pure_east),  # NE
                    (pure_south, pure_west),  # SW
                    (pure_south, pure_east),  # SE
                ]
                state = [previous_universe[nb] for nb in neighbors]
                alive_count = len([s for s in state if s])
                current_cell = previous_universe[j, i]
                universe[j, i] = (current_cell and alive_count in self.survive_rule) or (
                    not current_cell and alive_count in self.birth_rule
                )
        self.universe = universe
        if self.visualize:
            self.img.set_data(self.universe)
            last_time = perf_counter_ns()
            diff = last_time - self.last_time
            self.text.set_text(
                self.text_template.format(
                    self.generation,
                    round(1e3 / self.fps, 2),
                    round(self.fps).astype(int),
                    round(diff / 1e6, 2),
                    round(1e9 // diff).astype(int),
                )
            )
            self.last_time = last_time
        self.generation += 1
        if self.visualize:
            return self.img, self.text

    def run(self):
        if self.visualize:
            _ = FuncAnimation(self.fig, self.update, interval=int(1 / self.fps * 1000), init_func=self.init)
            plt.show()
        else:
            last_gen = self.generation
            while True:
                self.update(0)
                if ((perf_counter_ns() - self.last_time) / 1e9) >= 1:
                    print(f"{self.generation - last_gen} gen/s")
                    last_gen = self.generation
                    self.last_time = perf_counter_ns()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--size", help="Length of the square lattice, must be an integer power of 2", type=int, default=128
    )
    parser.add_argument("-t", "--threshold", help="Random initializer threshold", type=float, default=0.93)
    parser.add_argument("-r", "--rulestring", help="Rulestring to use. Uses CGOL default (B3/S23)", default="B3/S23")
    parser.add_argument("--fps", help="Requested FPS", type=float, default=23.976)
    args = parser.parse_args()

    if log2(args.size) % 1 != 0:
        raise ValueError(f"`--size` must be an integer power of 2 (got {args.size})")

    birth_string, survive_string = args.rulestring.split("/")
    birth_string = birth_string.strip("B")
    survive_string = survive_string.strip("S")
    birth_rule = [int(b) for b in birth_string]
    survive_rule = [int(s) for s in survive_string]

    Game(
        size=args.size,
        threshold=args.threshold,
        fps=args.fps,
        birth_rule=birth_rule,
        survive_rule=survive_rule,
    ).run()
