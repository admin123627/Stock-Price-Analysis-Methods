# Finance Data Science Toolkit

Welcome to the **Finance Data Science Toolkit**, a collection of Python scripts and notebooks focused on applying machine learning and statistical techniques to financial data.  

This repository contains:

- **`stock_svr.py`**  
  A command-line tool that:
  1. Fetches one year of daily stock data for a user-specified ticker.  
  2. Trains Support Vector Regression (SVR) models (Linear and RBF kernels) on the “Open” price.  
  3. Plots historical prices and model fits, extending predictions to a user-defined date with a dashed line.  
  4. Annotates the predicted price at the end date.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `scikit-learn`  
- `yfinance`

Install dependencies via pip:

```bash
pip install numpy pandas matplotlib scikit-learn yfinance
