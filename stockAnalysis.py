import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf  
import datetime

dates = []
prices = []

def analyze_and_plot(ticker: str, pred_date: datetime.datetime) -> None:
    """
    Download historical data for `ticker`, train SVR models, and plot
    both the historical "Open" prices and the SVR predictions up to `pred_date`,
    with the extension line in a different color and annotated prediction point.
    """
    # Download historical data (past year)
    print(f"Downloading {ticker} data for the past year…")
    df = yf.download(
        ticker,
        period="1y",
        auto_adjust=False,
        progress=False
    )
    if df.empty:
        print("No data fetched—check ticker or internet connection.")
        return

    # Extract dates and open prices
    dates = df.index.to_pydatetime()    # numpy array of datetime.datetime
    prices = df["Open"].to_numpy()    # shape (n_samples,)
    y = prices.ravel()                  # ensure 1D array (n_samples,)

    # Numeric conversion for regression
    X_hist = mdates.date2num(dates).reshape(-1, 1)

    # Train SVR models
    models = {
        "Linear": SVR(kernel="linear", C=1e3),
        "RBF":    SVR(kernel="rbf",    C=1e3, gamma=0.1),
    }
    for name, model in models.items():
        print(f"Training {name} SVR…")
        model.fit(X_hist, y)

    # Prepare extension range from last historical date to pred_date
    last_num = X_hist[-1, 0]
    target_num = mdates.date2num(pred_date)
    X_ext = np.linspace(last_num, target_num, 50)
    dates_ext = mdates.num2date(X_ext)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(dates, prices, s=10, color="black", label="Historical Data")

    # Colors for base fit and extension
    base_colors = {"Linear": "green", "RBF": "red"}
    ext_colors  = {"Linear": "lime",  "RBF": "orange"}

    # Plot historical fit and extension
    for name, model in models.items():
        # historical fit
        y_hist_pred = model.predict(X_hist)
        ax.plot(dates, y_hist_pred, color=base_colors[name], label=f"{name} Fit")
        # extension fit
        y_ext_pred = model.predict(X_ext.reshape(-1, 1))
        ax.plot(dates_ext, y_ext_pred, linestyle="--", color=ext_colors[name],
                label=f"{name} Extension")

    # Plot and annotate prediction point
    for name, model in models.items():
        y_target = model.predict([[target_num]])[0]
        ax.scatter(pred_date, y_target, color=base_colors[name], marker="X", s=100)
        ax.annotate(
            f"({pred_date.date()}, {y_target:.2f})",
            xy=(pred_date, y_target),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color=base_colors[name]
        )

    # Format x-axis with month ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    ax.set_title(f"SVR on {ticker} Open Price with Extension to {pred_date.date()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Open Price (USD)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    # User inputs
    ticker = input("Enter ticker symbol (e.g. INTC): ").strip().upper()
    pred_date_str = input("Enter prediction date (YYYY-MM-DD): ").strip()
    try:
        pred_date = datetime.datetime.strptime(pred_date_str, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    analyze_and_plot(ticker, pred_date)


if __name__ == "__main__":
    main()
