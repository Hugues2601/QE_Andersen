import numpy as np
import torch
from config import CONFIG
import matplotlib.pyplot as plt

class FSBlackScholes:
    def __init__(self, S0, k, T1, T2, r, sigma):
        self.S0 = torch.tensor([S0], device=CONFIG.device, dtype=torch.float32, requires_grad=True)
        self.k = torch.tensor(k, device=CONFIG.device,dtype=torch.float32)
        self.T1 = torch.tensor([T1],device=CONFIG.device, dtype=torch.float32)
        self.T2 = torch.tensor(T2,device=CONFIG.device, dtype=torch.float32)
        self.r = torch.tensor([r],device=CONFIG.device, dtype=torch.float32)
        self.sigma = sigma

    def _d1_d2(self):
        tau = self.T2 - self.T1
        d1 = (torch.log(1 / self.k) + (self.r + 0.5 * self.sigma ** 2) * tau) / (
                    self.sigma * torch.sqrt(tau))
        d2 = d1 - self.sigma * torch.sqrt(tau)
        return d1, d2

    def price(self):
        tau = self.T2 - self.T1
        d1, d2 = self._d1_d2()
        price = self.S0 * (torch.distributions.Normal(0, 1).cdf(d1) - self.k * torch.exp(-self.r * tau) * torch.distributions.Normal(0, 1).cdf(d2))
        return price

    def compute_greek(self, greek_name, batch=False):
        self.sigma = torch.tensor(self.sigma, device=CONFIG.device, dtype=torch.float32, requires_grad=True)

        greeks = {
            "delta": self.S0,
            "vega": self.sigma,
            "rho": self.r,
            "theta": 0.0,
            "gamma": self.S0,
            "vanna": (self.S0, self.sigma),
            "volga": self.sigma,
        }

        variable = greeks[greek_name]
        price = self.price()

        if batch:
            price = price.sum()

        if greek_name == "volga":
            first_derivative, = torch.autograd.grad(price, self.sigma, create_graph=True)
            second_derivative, = torch.autograd.grad(first_derivative.sum() if batch else first_derivative, self.sigma)
            volga = 2 * first_derivative + 4 * self.sigma * second_derivative
            return volga if batch else volga.item()

        if isinstance(variable, tuple):
            var1, var2 = variable
            first_derivative, = torch.autograd.grad(price, var1, create_graph=True)
            second_derivative, = torch.autograd.grad(first_derivative.sum() if batch else first_derivative, var2)
            return second_derivative if batch else second_derivative.item()

        else:
            derivative, = torch.autograd.grad(price, variable)

            if greek_name == "vega":
                adjusted_vega = derivative
                return adjusted_vega if batch else adjusted_vega.item()

            return derivative if batch else derivative.item()

if __name__ == "__main__":

    k = np.linspace(0.2, 2, 100)
    k_liste = []
    for element in k:
        price = FSBlackScholes(5667.65, element, 0.75, 1.5, 0.03927, 0.2)
        vega = price.compute_greek("vega")
        k_liste.append(vega)
    plt.plot(k, k_liste)
    plt.show()