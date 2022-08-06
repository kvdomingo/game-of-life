from random import SystemRandom
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import array


class Game:
    class State:
        def __init__(self, **kwargs):
            self.west = kwargs.get("west") or False
            self.east = kwargs.get("east") or False
            self.north = kwargs.get("north") or False
            self.south = kwargs.get("south") or False
            self.northwest = kwargs.get("northwest") or False
            self.northeast = kwargs.get("northeast") or False
            self.southwest = kwargs.get("southwest") or False
            self.southeast = kwargs.get("southeast") or False

        def dict(self):
            return dict(
                west=self.west,
                east=self.east,
                north=self.north,
                south=self.south,
                northwest=self.northwest,
                northeast=self.northeast,
                southwest=self.southwest,
                southeast=self.southeast,
            )

        def count_alive(self):
            return len([s for s in self.dict().values() if s])

    def __init__(self, size: int = 50, threshold: float = 0.5):
        random = SystemRandom()
        plt.style.use("seaborn")
        plt.rcParams.update(
            {
                "figure.figsize": (7, 7),
                "figure.dpi": 100,
            }
        )
        random.seed(42)
        self.generation = 1
        self.N = size
        self.universe = array([random.random() > threshold for _ in range(size**2)]).reshape((size, size))
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.img = self.ax.imshow(self.universe, cmap="gray", animated=True)
        self.text_template = "Generation: %i\n%i FPS"
        self.text = self.ax.text(0.025, 0.025, "", transform=self.ax.transAxes, fontsize=12, color="c")
        self.last_time = perf_counter()
        self.main()

    def init(self):
        self.ax.set_xlim(0, self.N)
        self.ax.set_ylim(0, self.N)
        self.ax.axis("off")
        self.ax.grid(False)
        self.text.set_text("")
        return self.img, self.text

    def update(self, _):
        universe = self.universe
        for j in range(self.N):
            for i in range(self.N):
                state = self.State()
                if 0 < i < self.N - 1:
                    state.west = universe[j, i - 1]
                    state.east = universe[j, i + 1]
                if 0 < j < self.N - 1:
                    state.north = universe[j - 1, i]
                    state.south = universe[j + 1, i]
                if 0 < i < self.N - 1 and 0 < j < self.N - 1:
                    state.northwest = universe[j - 1, i - 1]
                    state.northeast = universe[j - 1, i + 1]
                    state.southwest = universe[j + 1, i - 1]
                    state.southeast = universe[j + 1, i + 1]

                universe[j, i] = int(
                    (universe[j, i] and 2 <= state.count_alive() <= 3)
                    or (not universe[j, i] and state.count_alive() == 3)
                )
        self.universe = universe
        self.img.set_data(self.universe)
        last_time = perf_counter()
        self.text.set_text(self.text_template % (self.generation, round(1 / (last_time - self.last_time))))
        self.last_time = last_time
        self.generation += 1
        return self.img, self.text

    def main(self):
        _ = FuncAnimation(self.fig, self.update, self.N, interval=int(1 / 60 * 1000), init_func=self.init, blit=True)
        plt.show()


if __name__ == "__main__":
    Game(size=50, threshold=0.93)
