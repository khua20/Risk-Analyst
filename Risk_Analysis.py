import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define your API key and stock symbols
api_key = "Enter-Your-API-Key-Here"
symbols = [
    "AAPL",
    "MSFT",
    "TSLA",
    "GOOGL",
    "TLT",
]  # alphavantage free version only allows 5 calls per minute & 100 calls per day

# Define the risk-free rate
risk_free_rate = 0.02  # Adjust as needed

# Create an empty DataFrame to store daily price data for all assets
portfolio_data = pd.DataFrame()

# Fetch historical data for all assets
for symbol in symbols:
    endpoint = "https://www.alphavantage.co/query?"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if "Time Series (Daily)" in data:
            daily_data = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(daily_data, orient="index")
            # Convert columns to numeric
            df["1. open"] = pd.to_numeric(df["1. open"])
            df["2. high"] = pd.to_numeric(df["2. high"])
            df["3. low"] = pd.to_numeric(df["3. low"])
            df["4. close"] = pd.to_numeric(df["4. close"])
            df["5. volume"] = pd.to_numeric(df["5. volume"])

            # Drop rows with missing or NaN values
            df = df.dropna()

            # Calculate daily returns
            df["Daily Return"] = df["4. close"].pct_change()

            df.index = pd.to_datetime(df.index)
            portfolio_data = pd.concat([portfolio_data, df["4. close"]], axis=1)
        else:
            print(
                f"Error: 'Time Series (Daily)' key not found in API response for {symbol}."
            )
    else:
        print(
            f"Failed to retrieve data for {symbol}. Status code: {response.status_code}"
        )

# Rename columns with symbols
portfolio_data.columns = symbols

# Calculate daily returns for each asset
returns = portfolio_data.pct_change()

# Calculate mean daily returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Define the confidence level (e.g., 95%)
confidence_level = 0.95

# # Calculate VaR using historical simulation
# returns = portfolio_data.pct_change().dropna()
# initial_investment = 1000000  # Replace with your initial investment amount
# portfolio_value = initial_investment * (1 + returns).cumprod()
# portfolio_value = portfolio_value.fillna(initial_investment)
# portfolio_returns = portfolio_value.pct_change()
# var = portfolio_returns.quantile(1 - confidence_level)
# var_dollar = initial_investment * var

# # Convert var to a scalar value (float)
# var_value = var.values[0]

# print(f"Portfolio VaR at {confidence_level*100:.2f}% confidence: {var_value*100:.2f}%")

# Number of iterations for Monte Carlo simulation
num_simulations = 10000

# Initialize lists to store simulation results
results = []

for _ in range(num_simulations):
    # Randomly generate portfolio weights
    weights = np.random.random(len(symbols))
    weights /= np.sum(weights)

    # Expected portfolio return
    port_return = np.sum(weights * mean_returns) * 252  # Annualized

    # Expected portfolio volatility
    port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    # Sharpe ratio
    sharpe_ratio = (port_return - risk_free_rate) / port_stddev

    results.append([port_return, port_stddev, sharpe_ratio, weights])

# Convert results to a DataFrame
results_df = pd.DataFrame(
    results, columns=["Return", "Volatility", "Sharpe Ratio", "Weights"]
)

# Find portfolio with the highest Sharpe ratio
max_sharpe_port = results_df.iloc[results_df["Sharpe Ratio"].idxmax()]

# Find portfolio with the lowest volatility
min_volatility_port = results_df.iloc[results_df["Volatility"].idxmin()]

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(
    results_df["Volatility"],
    results_df["Return"],
    c=results_df["Sharpe Ratio"],
    cmap="YlGnBu",
    marker="o",
)
plt.title("Efficient Frontier")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.colorbar(label="Sharpe Ratio")
plt.scatter(
    max_sharpe_port["Volatility"],
    max_sharpe_port["Return"],
    marker="*",
    color="r",
    s=100,
    label="Max Sharpe Ratio",
)
plt.scatter(
    min_volatility_port["Volatility"],
    min_volatility_port["Return"],
    marker="*",
    color="g",
    s=100,
    label="Min Volatility",
)
plt.legend()
plt.show()

# Print optimal asset allocation
print("Optimal Asset Allocation (Max Sharpe Ratio Portfolio):")
for symbol, weight in zip(symbols, max_sharpe_port["Weights"]):
    print(f"{symbol}: {weight * 100:.2f}%")
