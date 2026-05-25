# Multi-Armed Bandit Problem Simulator

This project is an implementation and testing environment for the classic **Multi-Armed Bandit** problem in Reinforcement Learning. The application allows for the simulation and efficiency comparison of three popular decision-making algorithms in a stationary environment with a normal (Gaussian) distribution.

The main goal of the program is to optimize the selection of the bandit's arms to maximize the cumulative reward and minimize the so-called **Regret**, which is the loss resulting from not choosing the optimal arm at each step.

---

## 🧠 Implemented Algorithms

The project implements three fundamental approaches to balancing Exploration and Exploitation:

1. **Epsilon-Greedy (ε-Greedy)**:
* Chooses the best-known arm with a probability of `1 - ε`.
* With a probability of `ε`, it performs random exploration of the environment.
* The `ε` value is configurable from the settings files.


2. **UCB (Upper Confidence Bound)**:
* Implements the principle of *"optimism in the face of uncertainty"*.
* Selects an arm based on the average reward received so far, plus a confidence interval.
* Automatically reduces the exploration priority for arms that have already been checked multiple times.


3. **Thompson Sampling**:
* A Bayesian approach utilizing sampling from the posterior distribution.
* Due to the continuous, Gaussian nature of the rewards, the algorithm uses the conjugate prior of the **Normal-Inverse-Gamma (NIG)** distribution.
* Dynamically updates hyperparameters (`μ`, `ν`, `α`, `β`) based on observed rewards.



---

## 📁 Project Structure

The project is designed in a modular way, with a clear separation of the environment logic from the algorithms:

```text
├── bandit/
│   └── bandit.py          # Environment definition (arms, means, standard deviations, reward sampling)
├── algorithms/
│   ├── algorithm.py       # Abstract base class for algorithms
│   ├── e_greedy.py        # Epsilon-Greedy algorithm implementation
│   ├── ucb.py             # UCB algorithm implementation
│   └── thompson.py        # Thompson Sampling implementation with NIG update
├── data/
│   ├── 01.json            # Test environment configuration #1 (4 arms)
│   └── 02.json            # Test environment configuration #2 (9 arms)
├── settings.json          # Global simulation settings (number of games, epsilon value)
└── main.py                # Main runner script and results visualization

```

---

## 🛠️ Requirements and Installation

The program requires Python 3.8 or newer and the installation of numerical and graphical libraries.

1. Clone the repository or download the project files.
2. Install the required packages using the `pip` package manager:

```bash
pip install numpy matplotlib

```

---

## ⚙️ Simulation Configuration

### General Settings (`settings.json`)

Global launch parameters are defined in this file:

* `epsilon`: exploration coefficient for the `ε-Greedy` algorithm.
* `games`: total number of rounds (iterations/arm pulls) that make up one simulation.

```json
{
    "epsilon": 0.1,
    "games": 500
}

```

### Environment Definitions

JSON files in the `data/` folder define the reward structure for the multi-armed bandit:

* `arms`: number of available arms.
* `mean`: expected mean reward value for each arm.
* `std`: standard deviation of the reward (Gaussian noise) for each arm.

---

## 🚀 Running the Program

To run the simulation and compare the algorithms, execute the command:

```bash
python main.py -f 01.json

```

### Expected Console Output

Upon completion of the simulation, the program will print a summary of the obtained rewards in the terminal, as well as the result of a theoretical "clairvoyant" (choosing exclusively the best arm throughout the game):

```text
Rewards obtained by epsilon Greedy algorithm: 524.321948210342
Rewards obtained by UCB algorithm: 612.871239847102
Rewards obtained by Thompson sampling: 634.192048123951
Reward for clairvoyance: 650.0

```

### Results Visualization

The program will automatically generate and display a chart showing the **Total Regret** for each algorithm over time.

* *The flatter the chart (lower slope in later phases), the faster the algorithm identified the optimal arm and stopped wasting trials on worse options.*
