import numpy.typing as npt
import numpy as np

from algorithms.algorithm import Algorithm


class Thompson(Algorithm):
    alpha: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    mu: npt.NDArray[np.float64]
    nu: npt.NDArray[np.float64]

    def __init__(self, bandit, games=300):
        super().__init__(bandit, games)
        self.mu = np.zeros(self.arms)
        self.nu = np.ones(self.arms)
        self.alpha = np.ones(self.arms)
        self.beta = np.ones(self.arms)

    def play(self) -> float:

        for i in range(self.games):
            arm = self.select_arm()
            reward = self.bandit.pull(arm)

            self.update(arm, reward)

            if i != 0:
                self.regret[i] = self.regret[i - 1]
            self.regret[i] += self.bandit.best - reward
            self.total_reward += reward

        return self.total_reward

    def select_arm(self) -> int:
        sampled_vars = 1.0 / np.random.gamma(shape=self.alpha, scale=1.0 / self.beta)
        sampled_means = np.random.normal(self.mu, np.sqrt(sampled_vars / self.nu))

        return np.argmax(sampled_means)

    def update(self, arm: int, reward: float) -> None:
        mu_old, nu_old = self.mu[arm], self.nu[arm]
        alpha_old, beta_old = self.alpha[arm], self.beta[arm]

        # Obliczenia aktualizacyjne (wzory NIG)
        nu_new = nu_old + 1
        alpha_new = alpha_old + 0.5

        # Średnia ważona
        mu_new = (nu_old * mu_old + reward) / nu_new

        # Beta (suma kwadratów błędów + poprawka)
        diff = reward - mu_old
        beta_new = beta_old + (nu_old / (2 * nu_new)) * (diff**2)

        # Zapisanie
        self.mu[arm] = mu_new
        self.nu[arm] = nu_new
        self.alpha[arm] = alpha_new
        self.beta[arm] = beta_new
