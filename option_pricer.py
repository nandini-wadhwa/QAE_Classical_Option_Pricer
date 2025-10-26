import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from qiskit_finance.applications import (
    EuropeanCallPricing, EuropeanPutPricing,
    AsianCallPricing, AsianPutPricing,
    BarrierCallPricing, BarrierPutPricing
)
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit import Aer
from qiskit.utils import QuantumInstance
import plotly.graph_objects as go
import time

# Title and description
st.title("🚀 Ultimate Options Pricer: Quantum vs. Classical")
st.markdown("Explore all option types with comparative pricing, visuals, and insights! Compare Black-Scholes, Monte Carlo, and Quantum methods.")

# Sidebar: User inputs
st.sidebar.header("📊 Configure Your Option")
option_type = st.sidebar.selectbox(
    "What do you want to check?",
    ["European Call", "European Put", "Asian Call", "Asian Put", "Barrier Call (Up-and-Out)", "Barrier Put (Down-and-In)", "Bull Call Spread", "Bear Put Spread", "Straddle", "Strangle"]
)

# Common parameters
S = st.sidebar.slider("Spot Price (S)", 50, 200, 100, help="Current stock price")
T = st.sidebar.slider("Maturity (T, years)", 0.1, 5.0, 1.0, help="Time to expiration")
r = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.05, help="Annual risk-free rate")
q = st.sidebar.slider("Dividend Yield (q)", 0.0, 0.1, 0.0, help="Dividend yield")
sigma = st.sidebar.slider("Volatility (σ)", 0.1, 1.0, 0.2, help="Volatility guess")
num_paths = st.sidebar.slider("Monte Carlo Paths", 1000, 50000, 10000, help="More paths = better accuracy but slower")
shots = st.sidebar.slider("Quantum Shots", 500, 5000, 1000, help="Quantum simulation samples")

# Option-specific parameters
params = {}
if option_type in ["European Call", "European Put"]:
    params['K'] = st.sidebar.slider("Strike Price (K)", 50, 200, 105)
elif option_type in ["Asian Call", "Asian Put"]:
    params['K'] = st.sidebar.slider("Strike Price (K)", 50, 200, 105)
    params['averaging'] = st.sidebar.selectbox("Averaging Type", ["Arithmetic", "Geometric"], help="How to average the price path")
elif option_type in ["Barrier Call (Up-and-Out)", "Barrier Put (Down-and-In)"]:
    params['K'] = st.sidebar.slider("Strike Price (K)", 50, 200, 105)
    params['B'] = st.sidebar.slider("Barrier Level (B)", 50, 200, 120, help="Barrier price for knock-out/in")
elif option_type == "Bull Call Spread":
    params['K1'] = st.sidebar.slider("Lower Strike (K1, Buy Call)", 50, 150, 100)
    params['K2'] = st.sidebar.slider("Higher Strike (K2, Sell Call)", 100, 200, 110)
    if params['K1'] >= params['K2']:
        st.sidebar.error("K1 must be less than K2!")
elif option_type == "Bear Put Spread":
    params['K1'] = st.sidebar.slider("Higher Strike (K1, Buy Put)", 100, 200, 110)
    params['K2'] = st.sidebar.slider("Lower Strike (K2, Sell Put)", 50, 150, 100)
    if params['K1'] <= params['K2']:
        st.sidebar.error("K1 must be greater than K2!")
elif option_type == "Straddle":
    params['K'] = st.sidebar.slider("Strike Price (K)", 50, 200, 105, help="Call and Put at same strike")
elif option_type == "Strangle":
    params['K1'] = st.sidebar.slider("Call Strike (K1)", 50, 150, 100)
    params['K2'] = st.sidebar.slider("Put Strike (K2)", 100, 200, 110)

# Pricing functions
def black_scholes(S, K, T, r, q, sigma, is_call=True):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)

def monte_carlo_price(S, params, T, r, q, sigma, option_type, num_paths=10000):
    np.random.seed(42)
    dt = T / 252
    paths = np.zeros((num_paths, 252))
    paths[:, 0] = S
    for t in range(1, 252):
        z = np.random.normal(0, 1, num_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    
    if option_type in ["European Call", "European Put"]:
        K = params['K']
        payoffs = np.maximum(paths[:, -1] - K, 0) if "Call" in option_type else np.maximum(K - paths[:, -1], 0)
    elif option_type in ["Asian Call", "Asian Put"]:
        K = params['K']
        avg_prices = np.mean(paths, axis=1) if params['averaging'] == "Arithmetic" else np.exp(np.mean(np.log(paths), axis=1))
        payoffs = np.maximum(avg_prices - K, 0) if "Call" in option_type else np.maximum(K - avg_prices, 0)
    elif option_type == "Barrier Call (Up-and-Out)":
        K, B = params['K'], params['B']
        max_prices = np.max(paths, axis=1)
        payoffs = np.where(max_prices < B, np.maximum(paths[:, -1] - K, 0), 0)
    elif option_type == "Barrier Put (Down-and-In)":
        K, B = params['K'], params['B']
        min_prices = np.min(paths, axis=1)
        payoffs = np.where(min_prices <= B, np.maximum(K - paths[:, -1], 0), 0)
    elif option_type == "Bull Call Spread":
        K1, K2 = params['K1'], params['K2']
        payoffs = np.maximum(paths[:, -1] - K1, 0) - np.maximum(paths[:, -1] - K2, 0)
    elif option_type == "Bear Put Spread":
        K1, K2 = params['K1'], params['K2']
        payoffs = np.maximum(K1 - paths[:, -1], 0) - np.maximum(K2 - paths[:, -1], 0)
    elif option_type == "Straddle":
        K = params['K']
        payoffs = np.maximum(paths[:, -1] - K, 0) + np.maximum(K - paths[:, -1], 0)
    elif option_type == "Strangle":
        K1, K2 = params['K1'], params['K2']
        payoffs = np.maximum(paths[:, -1] - K1, 0) + np.maximum(K2 - paths[:, -1], 0)
    
    return np.exp(-r*T) * np.mean(payoffs)

def quantum_price(S, params, T, r, q, sigma, option_type, num_qubits=5, shots=1000):
    mu = (r - q - sigma**2/2) * T
    var = sigma**2 * T
    uncertainty_model = LogNormalDistribution(num_qubits, mu=mu, sigma=np.sqrt(var), bounds=(0, 2*S))
    
    if option_type == "European Call":
        pricer = EuropeanCallPricing(num_qubits, uncertainty_model, params['K'], r, T)
    elif option_type == "European Put":
        pricer = EuropeanPutPricing(num_qubits, uncertainty_model, params['K'], r, T)
    elif option_type == "Asian Call":
        pricer = AsianCallPricing(num_qubits, uncertainty_model, params['K'], r, T, arithmetic=params['averaging']=="Arithmetic")
    elif option_type == "Asian Put":
        pricer = AsianPutPricing(num_qubits, uncertainty_model, params['K'], r, T, arithmetic=params['averaging']=="Arithmetic")
    elif option_type == "Barrier Call (Up-and-Out)":
        pricer = BarrierCallPricing(num_qubits, uncertainty_model, params['K'], params['B'], r, T, up=True, out=True)
    elif option_type == "Barrier Put (Down-and-In)":
        pricer = BarrierPutPricing(num_qubits, uncertainty_model, params['K'], params['B'], r, T, up=False, out=False)
    else:
        # For spreads/straddles, approximate as sum/difference (not native in Qiskit, so use MC fallback)
        return monte_carlo_price(S, params, T, r, q, sigma, option_type, num_paths=shots)
    
    backend = Aer.get_backend('aer_simulator')
    qi = QuantumInstance(backend, shots=shots)
    result = pricer.estimate(qi)
    return result['estimation']

# Compute prices
with st.spinner("Calculating prices..."):
    progress_bar = st.progress(0)
    
    # BS Price (analytical where possible, else MC approximation)
    if option_type in ["European Call", "European Put"]:
        bs_price = black_scholes(S, params['K'], T, r, q, sigma, "Call" in option_type)
    elif option_type in ["Bull Call Spread", "Bear Put Spread", "Straddle", "Strangle"]:
        # Sum/difference of BS
        if option_type == "Bull Call Spread":
            bs_price = black_scholes(S, params['K1'], T, r, q, sigma, True) - black_scholes(S, params['K2'], T, r, q, sigma, True)
        elif option_type == "Bear Put Spread":
            bs_price = black_scholes(S, params['K1'], T, r, q, sigma, False) - black_scholes(S, params['K2'], T, r, q, sigma, False)
        elif option_type == "Straddle":
            bs_price = black_scholes(S, params['K'], T, r, q, sigma, True) + black_scholes(S, params['K'], T, r, q, sigma, False)
        elif option_type == "Strangle":
            bs_price = black_scholes(S, params['K1'], T, r, q, sigma, True) + black_scholes(S, params['K2'], T, r, q, sigma, False)
    else:
        bs_price = monte_carlo_price(S, params, T, r, q, sigma, option_type, num_paths)  # Approximation
    
    progress_bar.progress(33)
    mc_price = monte_carlo_price(S, params, T, r, q, sigma, option_type, num_paths)
    progress_bar.progress(66)
    q_price = quantum_price(S, params, T, r, q, sigma, option_type, shots=shots)
    progress_bar.progress(100)

# Comparative Table
st.header("📈 Comparative Pricing Table")
comparison_df = pd.DataFrame({
    "Method": ["Black-Scholes (Analytical/Approx)", "Classical Monte Carlo", "Quantum Computing (Qiskit)"],
    "Price": [f"${bs_price:.2f}", f"${mc_price:.2f}", f"${q_price:.2f}"]
})
st.table(comparison_df)

# Tabs for visuals
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Payoff Diagram", "📈 Price vs. Volatility", "🔄 Monte Carlo Convergence", "⚖️ Error Comparison", "📉 Sensitivity Analysis"])

with tab1:
    st.subheader("Payoff Diagram (Illustrative)")
    stock_range = np.linspace(50, 150, 100)
    if option_type in ["European Call", "European Put"]:
        payoff = np.maximum(stock_range - params['K'], 0) if "Call" in option_type else np.maximum(params['K'] - stock_range, 0)
    elif option_type == "Bull Call Spread":
        payoff = np.maximum(stock_range - params['K1'], 0) - np.maximum(stock_range - params['K2'], 0)
    elif option_type == "Bear Put Spread":
        payoff = np.maximum(params['K1'] - stock_range, 0) - np.maximum(params['K2'] - stock_range, 0)
    elif option_type == "Straddle":
        payoff = np.maximum(stock_range - params['K'], 0) + np.maximum(params['K'] - stock_range, 0)
    elif option_type == "Strangle":
        payoff = np.maximum(stock_range - params['K1'], 0) + np.maximum(params['K2'] - stock_range, 0)
    else:
        payoff = np.zeros_like(stock_range)  # Placeholder for path-dependent
        st.info("Payoff for Asians/Barriers is path-dependent; this is a simplified European-like illustration.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_range, y=payoff, mode='lines', name=option_type, line=dict(color='green')))
    fig.update_layout(title=f"Payoff at Maturity for {option_type}", xaxis_title="Stock Price", yaxis_title="Payoff")
    st.plotly_chart(fig)

with tab2:
    st.subheader("Price vs. Volatility Comparison")
    vol_range = np.linspace(0.1, 0.5, 10)
    bs_prices = [black_scholes(S, params.get('K', 100), T, r, q, v, "Call" in option_type) if "European" in option_type else monte_carlo_price(S, params, T, r, q, v, option_type, 5000) for v in vol_range]
    mc_prices = [monte_carlo_price(S, params, T, r, q, v, option_type, 5000) for v in vol_range]
    q_prices = [quantum_price(S, params, T, r, q, v, option_type, shots=500) for v in vol_range]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_range, y=bs_prices, mode='lines', name='Black-Scholes', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=vol_range, y=mc_prices, mode='lines', name='Monte Carlo', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=vol_range, y=q_prices, mode='lines', name='Quantum', line=dict(color='red', dash='dash')))
    fig.update_layout(title="Price Sensitivity to Volatility", xaxis_title="Volatility (σ)", yaxis_title="Price")
    st.plotly_chart(fig)

with tab3:
    st.subheader("Monte Carlo Convergence")
    np.random.seed(42)
    paths = 10000
    payoffs = []
    for i in range(1, paths+1):
        z = np.random.normal(0, 1)
        ST = S * np.exp((r - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
        if option_type in ["European Call", "European Put"]:
            payoff = max(ST - params['K'], 0) if "Call" in option_type else max(params['K'] - ST, 0)
        else:
            payoff = 0  # Simplified
        payoffs.append(np.exp(-r*T) * payoff)
    cumulative_avg = np.cumsum(payoffs) / np.arange(1, paths+1)
   
