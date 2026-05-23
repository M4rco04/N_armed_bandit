import json
import matplotlib.pyplot as plt
from typing import List

from bandit.bandit import Bandit
from algorithms.e_greedy import EpsilonGreedy
from algorithms.ucb import UCB
from algorithms.thompson import Thompson
from algorithms.algorithm import Algorithm

import argparse

def main():
    parser = argparse.ArgumentParser(description="Symulator N armed bandit")
    parser.add_argument("-f", "--file", help="Plik, z którego wczytywane są statystyki automatów")
    args = parser.parse_args()

    if not args.file:
        raise ValueError("Nie podano pliku")

    with open("settings.json", "r") as file:
        data = json.load(file)

    with open(f"data/{args.file}", "r") as file:
        statistics = json.load(file)

    EPSILON, GAMES = data["epsilon"], data["games"]
    MEAN, STD, ARMS = statistics["mean"], statistics["std"], statistics["arms"]

    bandit = Bandit(ARMS, MEAN, STD)
    greedy = EpsilonGreedy(bandit, GAMES, EPSILON)
    ucb = UCB(bandit, GAMES)
    thompson = Thompson(bandit, GAMES)

    print("Nagrody zdobyte przez algorytm epsilon Greedy:", greedy.play())
    print("Nagrody zdobyte przez algorytm UCB:", ucb.play())
    print("Nagrody zdobyte przez próbkowanie Thompsona:", thompson.play())
    print("Nagroda przy jasnowidzeniu:", bandit.best * GAMES)

    draw_plots(
        [greedy.regret, ucb.regret, thompson.regret],
        [f"Epsilon greedy {greedy.epsilon}", "UCB", "Thompson sampling"],
    )


def draw_plots(alogrithms: List[Algorithm], legends: List[str]) -> None:
    plt.figure(figsize=(10, 10))
    plt.title("Total regret during game")
    for i, rewards in enumerate(alogrithms):
        plt.plot(rewards, label=f"{legends[i]}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
