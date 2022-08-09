from random import SystemRandom
from time import perf_counter_ns

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import Axes, Figure, Text
from numpy import array, log2, ndarray, round, zeros


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
            self.fig.tight_layout()
            self.img = self.ax.imshow(self.universe, cmap="gray", origin="lower", animated=True)
            self.text_template = "\n".join(
                [
                    "gen: %i",
                    "req ms: %.2f (%i fps)",
                    "act ms: %.2f (%i fps)",
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
            pure_north = 0 if j + 1 > N - 1 else j + 1
            pure_south = N - 1 if j - 1 < 0 else j - 1
            for i in range(N):
                pure_east = 0 if i + 1 > N - 1 else i + 1
                pure_west = N - 1 if i - 1 < 0 else i - 1
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
                self.text_template
                % (self.generation, round(1e3 / self.fps, 2), self.fps, round(diff / 1e6, 2), (1e9 // diff))
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
    Game().run()
