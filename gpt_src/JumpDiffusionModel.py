import numpy as np

class JumpDiffusionModel:
    def __init__(self, S0, r, sigma, lambd, mu_J, T, num_steps):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.lambd = lambd
        self.mu_J = mu_J
        self.T = T
        self.num_steps = num_steps
        self.dt = T / num_steps

    def simulate(self):
        S = np.zeros(self.num_steps + 1)
        S[0] = self.S0

        for i in range(1, self.num_steps + 1):
            Z = np.random.normal(0, 1)
            J = np.random.poisson(self.lambd * self.dt)

            dS = (self.r - self.lambd * self.mu_J) * S[i - 1] * self.dt + self.sigma * S[i - 1] * np.sqrt(self.dt) * Z + J
            S[i] = S[i - 1] + dS

        return S