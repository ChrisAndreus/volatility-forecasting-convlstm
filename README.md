Quantitative Volatility Trading – Research to Code
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/PyTorch-1.9+-orange.svg
https://img.shields.io/badge/License-MIT-green.svg

📖 Overview
This repository contains a Python implementation inspired by two influential research papers in quantitative finance and machine learning:

"Recipe For Quantitative Trading With Machine Learning" – Daniel Bloch (2018, revised 2023)

"Deep Learning Based Dynamic Implied Volatility Surface" – Daniel Bloch & Arthur Book (2021)

The code translates the theoretical concepts from these papers into a working deep learning framework for forecasting volatility dynamics. It combines rough volatility modelling, ensemble learning, and spatiotemporal sequence forecasting using a ConvLSTM architecture.

🧠 What This Project Does
Generates synthetic volatility data with realistic rough‑vol dynamics (fractional Brownian noise, regime shifts) as described in the Bloch (2018) paper.

Applies PCA dimensionality reduction to compress the 40‑dimensional parameter space (5 SVI parameters × 8 maturities) into a smaller set of factors.

Implements a ConvLSTM network (convolutional LSTM) with encoder‑decoder structure for multi‑step ahead forecasting of the volatility surface – following the approach in Bloch & Book (2021).

Compares performance against classical benchmarks: HAR‑RV and Exponential Smoothing (from the “Recipe” paper).

Uses Monte Carlo Dropout for uncertainty estimation (concept from the deep learning literature, applied here).

Evaluates forecasts with both statistical (RMSE, MAE) and economic metrics (variance swap error).

Provides visualisations of forecast accuracy, skill decay, and uncertainty bands.

📚 Research Foundation
Paper 1: Recipe For Quantitative Trading With Machine Learning
Author: Daniel Bloch
Key concepts used in this code:

Multifractal nature of financial time series – the data generator includes long‑range dependence and regime shifts (Section 2.1).

Ensemble methods – the EnsembleClassifier combines TabM, Gradient Boosting, and Logistic Regression (Section 7).

Baseline models – HAR‑RV and exponential smoothing (Appendix 9.2.3) serve as benchmarks.

Kelly portfolio optimisation – implemented in kelly_portfolio and calculate_optimal_three_way_bets (Section 2.3).

Time series cross‑validation – walk‑forward simulation using TimeSeriesSplit (Section 2.4.4).

Paper 2: Deep Learning Based Dynamic Implied Volatility Surface
Authors: Daniel Bloch & Arthur Book
Key concepts used in this code:

Dimensionality reduction via PCA – reduces the 40‑dimension parameter space to 10 principal components (Section 2.2).

ConvLSTM architecture – the ConvLSTMCell and ConvLSTMNet classes implement the spatiotemporal forecasting model described in Section 3.2.2.

Encoder‑decoder forecasting – the model uses an encoding network to compress the input sequence and a decoding network to generate multi‑step predictions (Section 3.2.2).

Synthetic data generation – the RealisticVolDataGenerator mimics the rough‑vol dynamics of implied volatility surfaces (Section 2.1).

Evaluation metrics – both RMSE/MAE and economic metrics (variance swap error) are used (Section 4.3.2).

⚙️ Installation
Clone the repository:

bash
git clone https://github.com/yourusername/quantitative-volatility-trading.git
cd quantitative-volatility-trading
(Optional) Create a virtual environment:

bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies:

bash
pip install numpy pandas torch scikit-learn scipy matplotlib tqdm
🚀 Usage
Run the main script:

bash
python quantitative_volatility_trading.py
The script will:

Generate a synthetic volatility surface dataset (2000 days, 8 maturities).

Apply PCA to reduce dimensionality.

Train the ConvLSTM model on the first 70% of data.

Evaluate forecasts for horizons 1 to 60 days.

Display four diagnostic plots:

RMSE vs forecast horizon for all models

Variance swap error (economic metric)

Skill decay (Deep RMSE minus HAR‑RMSE)

Uncertainty bands for a single forecast path

Print summary metrics to the console.

All outputs are generated automatically – no external data files are required.

📁 Project Structure
text
.
├── quantitative_volatility_trading.py   # Main script
├── requirements.txt                      # Dependencies
├── README.md                              # This file
├── LICENSE                                # MIT License
└── images/                                 # Screenshots for README (optional)
    ├── rmse_plot.png
    ├── varswap_plot.png
    └── uncertainty_plot.png
🔬 Methodology
1. Synthetic Data Generation (RealisticVolDataGenerator)
Simulates daily SVI parameters (a, b, ρ, m, σ) for 8 maturities.

Uses fractional Brownian noise (H≈0.1–0.15) for a and b to capture rough volatility persistence.

Includes mean‑reversion and occasional regime shifts (e.g., correlation flips, volatility‑of‑volatility jumps).

2. Dimensionality Reduction (PCA)
Flattens the 5 × 8 = 40‑dimensional parameter space.

Retains the top 10 principal components (explaining ~95% variance).

3. ConvLSTM Model (ConvLSTMNet)
2 encoder layers + 2 decoder layers with hidden dimension 32.

Inputs: sequence of PCA components and forward prices (concatenated).

Trained to predict the delta (change) in PCA components.

Dropout is applied during training; during inference, MC dropout (30 forward passes) yields uncertainty estimates.

4. Benchmark Models (BaselineForecaster)
HAR‑RV: uses daily, weekly, and monthly average volatilities.

Exponential Smoothing + Linear Trend: combines exponentially weighted moving average with a recent linear trend.

5. Evaluation
Statistical: RMSE and MAE on PCA components.

Economic: variance swap error – RMSE on implied variance σ², relevant for trading.

Alignment: Hungarian algorithm corrects label switching among PCA components.

📊 Results (Example Output)
When run, the script produces plots similar to the ones below (actual outputs may vary):

RMSE vs Horizon	Variance Swap Error
https://images/rmse_plot.png	https://images/varswap_plot.png
Skill Decay	Uncertainty Bands
https://images/skill_plot.png	https://images/uncertainty_plot.png
Typical findings:

The ConvLSTM model outperforms HAR‑RV for horizons up to ~45 days.

Economic error (variance swap) grows roughly linearly with horizon.

Uncertainty bands expand naturally, reflecting model confidence.

📖 References
Bloch, D. (2018, revised 2023). Recipe For Quantitative Trading With Machine Learning. Working Paper.
Link to SSRN (if available)

Bloch, D., & Book, A. (2021). Deep Learning Based Dynamic Implied Volatility Surface. Working Paper.
Link to SSRN (if available)

Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. Quantitative Finance, 18(6), 933–949.

Shi, X., et al. (2015). Convolutional LSTM Network: A machine learning approach for precipitation nowcasting. Advances in Neural Information Processing Systems, 28.

📝 License
This project is licensed under the MIT License – see the LICENSE file for details.

🤝 Acknowledgements
The author thanks the researchers whose work made this implementation possible.

This project was developed as part of a personal learning journey to translate academic research into practical code, with assistance from AI tools to accelerate implementation and understanding.

📬 Contact
Created by Chris Andreus – feel free to reach out for questions, suggestions, or collaborations!
