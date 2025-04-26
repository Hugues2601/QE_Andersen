from abc import ABC, abstractmethod
import torch
from config import CONFIG
import matplotlib.pyplot as plt

class HestonModel(ABC):
    def __init__(self, S0, K, T, r, kappa, v0, theta, sigma, rho, type="call"):
        self.S0 = torch.tensor(S0, device=CONFIG.device, requires_grad=True)
        self.K = self._ensure_1d_tensor(torch.tensor(K, device=CONFIG.device))
        self.T = self._ensure_1d_tensor(torch.tensor(T, device=CONFIG.device))
        self.r = torch.tensor(r, device=CONFIG.device, requires_grad=True)
        self.kappa = torch.tensor(kappa, device=CONFIG.device)
        self.v0 = torch.tensor(v0, device=CONFIG.device, requires_grad=True)
        self.theta = torch.tensor(theta, device=CONFIG.device)
        self.sigma = torch.tensor(sigma, device=CONFIG.device, requires_grad=True)
        self.rho = torch.tensor(rho, device=CONFIG.device)
        self.type = type

    @abstractmethod
    def _heston_cf(self, phi):
        pass

    def _compute_integrals(self):

        # Vérification de la taille de K et T
        assert self.K.dim() == 1, "K doit être un tenseur 1D"
        assert self.T.dim() == 1, "T doit être un tenseur 1D"
        assert len(self.K) == len(self.T), "K et T doivent avoir la même taille"


        umax = 500
        n = 2000
        if n % 2 == 0:
            n += 1

        phi_values = torch.linspace(1e-5, umax, n, device=self.K.device)
        du = (umax - 1e-5) / (n - 1)

        phi_values = phi_values.unsqueeze(1).repeat(1, len(self.K))

        factor1 = torch.exp(-1j * phi_values * torch.log(self.K))
        denominator = 1j * phi_values


        cf1 = self._heston_cf(phi_values - 1j) / self._heston_cf(-1j)
        temp1 = factor1 * cf1 / denominator
        integrand_P1_values = 1 / torch.pi * torch.real(temp1)


        cf2 = self._heston_cf(phi_values)
        temp2 = factor1 * cf2 / denominator
        integrand_P2_values = 1 / torch.pi * torch.real(temp2)

        weights = torch.ones(n, device=self.K.device)
        weights[1:-1:2] = 4
        weights[2:-2:2] = 2
        weights *= du / 3
        weights = weights.unsqueeze(1).repeat(1, len(self.K))

        integral_P1 = torch.sum(weights * integrand_P1_values, dim=0)
        integral_P2 = torch.sum(weights * integrand_P2_values, dim=0)

        P1 = torch.tensor(0.5, device=self.K.device) + integral_P1
        P2 = torch.tensor(0.5, device=self.K.device) + integral_P2

        return P1, P2

    @abstractmethod
    def heston_price(self):
        pass

    @abstractmethod
    def compute_greek(self, greek_name):
        pass

    def _ensure_1d_tensor(self, tensor):
        """
        Assure que l'entrée est un tenseur 1D.
        Si l'entrée est un scalaire encapsulé dans un tenseur, elle sera transformée en 1D.
        """
        if tensor.dim() == 0:  # Si c'est un scalaire encapsulé
            return tensor.unsqueeze(0)  # Convertir en tenseur 1D
        return tensor










class ForwardStart(HestonModel):
    def __init__(self, S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
        super().__init__(S0=S0, K=k, T=T2, r=r, kappa=kappa, v0=v0, theta=theta, sigma=sigma, rho=rho)
        self.k = self._ensure_1d_tensor(torch.tensor(k, device=CONFIG.device))
        self.T0 = torch.tensor(T0, device=CONFIG.device, requires_grad=True)
        self.T1 = torch.tensor(T1, device=CONFIG.device)
        self.T2 = torch.tensor(T2, device=CONFIG.device)


    def _heston_cf(self, phi):
        # Ensure that phi is a torch tensor on the GPU
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.complex128, device=CONFIG.device)
        else:
            phi = phi.to(CONFIG.device).type(torch.complex128)


        S0 = self.S0.to(CONFIG.device).type(torch.float32)
        T0 = self.T0.to(CONFIG.device).type(torch.float32)
        T1 = self.T1.to(CONFIG.device).type(torch.float32)
        T2 = self.T2.to(CONFIG.device).type(torch.float32)
        r = self.r.to(CONFIG.device).type(torch.float32)

        tau = T2-T1

        delta = 4*self.kappa*self.theta/self.sigma**2
        little_c_bar = self.sigma**2/(4*self.kappa) * (1 - torch.exp(-self.kappa*(T1-T0)))
        kappa_bar = (4*self.kappa*self.v0*torch.exp(-self.kappa*(T1-T0))) / (self.sigma**2 * (1-torch.exp(-self.kappa*(T1-T0))))
        d = torch.sqrt((self.kappa-self.rho*self.sigma*1j*phi)**2 + self.sigma**2 * (phi**2 + 1j * phi))
        g = (self.kappa - self.rho*self.sigma*1j*phi-d)/(self.kappa-self.rho*self.sigma*1j*phi+d)

        A_bar = (
                self.r * 1j * phi * tau
                + (self.kappa * self.theta * tau / (self.sigma ** 2)) * (self.kappa - self.sigma * self.rho * 1j * phi - d)
                - (2 * self.kappa * self.theta / (self.sigma ** 2)) * torch.log((1.0 - g * torch.exp(-d * tau)) / (1.0 - g))
        )

        C_bar = (1-torch.exp(-d*tau))/(self.sigma**2 * (1-g*torch.exp(-d*tau))) * (self.kappa-self.rho*self.sigma*1j*phi - d)

        cf = torch.exp(A_bar + (C_bar * little_c_bar*kappa_bar)/(1 - 2*C_bar*little_c_bar)) * (1/(1-2*C_bar*little_c_bar))**(delta/2)
        return cf

    def heston_price(self):
        P1, P2 = self._compute_integrals()
        price = self.S0 * (P1 - self.k * torch.exp(-self.r * (self.T2 - self.T1)) * P2)
        return price

    def compute_greek(self, greek_name, batch=False):
        greeks = {
            "delta": self.S0,
            "vega": self.v0,
            "rho": self.r,
            "theta": self.T0,
            "gamma": self.S0,
            "vanna": (self.S0, self.v0),
            "volga": self.v0,
        }

        variable = greeks[greek_name]
        price = self.heston_price()

        if batch:
            price = price.sum()

        if greek_name == "volga":
            first_derivative, = torch.autograd.grad(price, self.v0, create_graph=True)
            second_derivative, = torch.autograd.grad(first_derivative.sum() if batch else first_derivative, self.v0)
            volga = 2 * first_derivative + 4 * self.v0 * second_derivative
            return volga if batch else volga.item()

        if isinstance(variable, tuple):
            var1, var2 = variable
            first_derivative, = torch.autograd.grad(price, var1, create_graph=True)
            second_derivative, = torch.autograd.grad(first_derivative.sum() if batch else first_derivative, var2)
            return second_derivative if batch else second_derivative.item()

        else:
            derivative, = torch.autograd.grad(price, variable)

            if greek_name == "vega":
                adjusted_vega =2 * torch.sqrt(self.v0) * derivative
                return adjusted_vega if batch else adjusted_vega.item()

            return derivative if batch else derivative.item()


