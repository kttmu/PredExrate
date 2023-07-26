import numpy as np

class StochasticVolatilityModel:
    def __init__(self, S0, r, kappa, theta, xi, T, num_steps):
        self.S0 = S0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.T = T
        self.num_steps = num_steps
        self.dt = T / num_steps

    def simulate(self):
        S = np.zeros(self.num_steps + 1)
        sigma = np.zeros(self.num_steps + 1)
        S[0] = self.S0
        sigma[0] = self.theta

        for i in range(1, self.num_steps + 1):
            Z = np.random.normal(0, 1)
            W = np.random.normal(0, 1)

            dS = self.r * S[i - 1] * self.dt + sigma[i - 1] * S[i - 1] * np.sqrt(self.dt) * W
            dSigma = self.kappa * (self.theta - sigma[i - 1]) * self.dt + self.xi * np.sqrt(sigma[i - 1] * self.dt) * Z

            S[i] = S[i - 1] + dS
            sigma[i] = sigma[i - 1] + dSigma

        return S, sigma
