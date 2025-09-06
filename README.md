# Project-1-European-Option-Pricing
European Option Pricing with Monte Carlo Simulation. Simulates 10,000 asset price trajectories using Geometric Brownian Motion, calculates call and put option payoffs, discounts them with the risk-free rate, and compares results with Black–Scholes. Includes trajectory plots for visualization.

#10,000 Simulations of an Asset price using Monte Carlo
import numpy as np 
import matplotlib.pyplot as plt 

#Parameters
#K : float - strike price
#S0 : float - initial stock price
#mu : float - drift
#sigma : float - volatility
#r : float - risk-free rate
#T : float - time horizon in years
#N : int - number of steps
#simulations : int - number of Monte Carlo simulations

def MonteCarlo(K,S0,mu,sigma,r,T,N,simulations):
    dt=T/N
    simulated_path=[]
    call_payoffs=[]
    put_payoffs=[]

    for _ in range(simulations):
        z=np.random.normal(0,1,N)
        increments=np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
        S_path=S0*np.cumprod(increments)
        simulated_path.append(S_path)
    
        S_T=S_path[-1]
        call_payoffs.append(max(S_T-K,0))
        put_payoffs.append(max(K-S_T,0))

    plt.figure(figsize=(10,6))
    for i in range(10):
        plt.plot(simulated_path[i])
        mean_path = np.mean(simulated_path, axis=0)
    plt.plot(mean_path, color='red', lw=2, label='Mean of Trajectories')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Simulated prices of the share", fontsize=12)
    plt.title("10 simulations of an Asset price using Monte Carlo", fontsize=14)
    plt.legend()
    plt.show()    
        
        
#Actualized prices of the call/put

    a_price_call=np.exp(-r*T)*np.mean(call_payoffs)
    a_price_put=np.exp(-r*T)*np.mean(put_payoffs)

    print("The call price is : ", round(a_price_call,3))
    print("The put price is : ", round(a_price_put,3))
    
#Comparison with Blach-Scholes' Model
    from scipy.stats import norm
    d1=(np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)

    BS_call=S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    BS_put=K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

    print("The Black-Scholes call price is : ", round(BS_call,3))
    print("The Black-Scholes put price is : ", round(BS_put,3))
