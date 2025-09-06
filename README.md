
    # 📊 Project 1 – European Option Pricing

This project simulates 10,000 asset price trajectories using **Geometric Brownian Motion** to price European call and put options. It compares the results with the **Black-Scholes formula** and includes trajectory plots for visualization.

---

## 🧮 Parameters

- `K` : Strike price  
- `S0` : Initial stock price  
- `mu` : Drift  
- `sigma` : Volatility  
- `r` : Risk-free rate  
- `T` : Time horizon (years)  
- `N` : Number of time steps  
- `simulations` : Number of Monte Carlo simulations  

---

## 🧠 Monte Carlo Simulation Code

```python
import numpy as np
import matplotlib.pyplot as plt

def MonteCarlo(K, S0, mu, sigma, r, T, N, simulations):
    dt = T / N
    simulated_path = []
    call_payoffs = []
    put_payoffs = []

    for _ in range(simulations):
        z = np.random.normal(0, 1, N)
        increments = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        S_path = S0 * np.cumprod(increments)
        simulated_path.append(S_path)

        S_T = S_path[-1]
        call_payoffs.append(max(S_T - K, 0))
        put_payoffs.append(max(K - S_T, 0))

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.plot(simulated_path[i])
    mean_path = np.mean(simulated_path, axis=0)
    plt.plot(mean_path, color='red', lw=2, label='Mean of Trajectories')
    plt.xlabel("Time")
    plt.ylabel("Simulated Share Price")
    plt.title("Monte Carlo Simulations")
    plt.legend()
    plt.show()

    # Discounted prices
    a_price_call = np.exp(-r * T) * np.mean(call_payoffs)
    a_price_put = np.exp(-r * T) * np.mean(put_payoffs)

    print("Call Option Price:", round(a_price_call, 3))
    print("Put Option Price:", round(a_price_put, 3))

