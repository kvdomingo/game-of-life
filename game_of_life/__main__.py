from argparse import ArgumentParser
from random import SystemRandom
from time import perf_counter

import cv2 as cv
import matplotlib.pyplot as plt
from numpy import array, floor, log2, ndarray


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
        self.universe: ndarray = initial_universe.astype("bool")
        self.last_time = perf_counter()
        self.last_fps = 0
        self.window = "game"

        if self.visualize:
            cv.namedWindow(self.window, cv.WINDOW_NORMAL)
            cv.resizeWindow(self.window, 1024, 1024)

    def evaluate_life(self, current_cell: bool, alive_count: int):
        return (current_cell and alive_count in self.survive_rule) or (
            not current_cell and alive_count in self.birth_rule
        )

    def update(self):
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
                universe[j, i] = self.evaluate_life(current_cell, alive_count)
        self.universe = universe
        self.last_fps = int(floor(1 / (perf_counter() - self.last_time)))
        self.last_time = perf_counter()
        self.generation += 1

    def run(self):
        if self.visualize:
            while True:
                self.update()
                cv.imshow(self.window, self.universe.astype("uint8") * 255)
                cv.setWindowTitle(self.window, f"{self.window} - {self.last_fps} FPS / Gen {self.generation}")
                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            cv.destroyAllWindows()
        else:
            last_gen = self.generation
            while True:
                self.update()
                if (perf_counter() - self.last_time) >= 1:
                    print(f"{self.generation - last_gen} gen/s")
                    last_gen = self.generation
                    self.last_time = perf_counter()


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
