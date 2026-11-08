import os
import numpy as np
import matplotlib.pyplot as plt
from data import load_series

plt.rcParams.update({'font.size': 12, "figure.dpi": 100})

YEAR = 365.25
YEARS_TO_DISPLAY = 5

os.makedirs("figures", exist_ok=True)


def autocorrelation(x, max_lag):
    x = x - np.mean(x)
    n = len(x)

    X = np.fft.rfft(x, n=2*n)
    corr = np.fft.irfft(X * np.conjugate(X), n=2*n)

    corr = corr[:max_lag + 1]
    corr = corr / corr[0]

    return corr


variables = {
    "tavg": "Mean daily temperature (°C)",
    "prcp": "Daily precipitation (mm)"
}

for var, label in variables.items():

    s = load_series(var)
    x = s.values.astype(float)

    if np.isnan(x).any():
        raise ValueError(f"Missing values in {var}")

    N = len(x)
    MAX_LAG = N - 1

    # Calculate autocorrelation for every daily lag
    acf = autocorrelation(x, MAX_LAG)
    lags = np.arange(len(acf))

    # Display only the first five years
    max_lag_display = min(int(YEARS_TO_DISPLAY * YEAR), MAX_LAG)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        lags[:max_lag_display + 1],
        acf[:max_lag_display + 1],
        lw=0.8,
        label="Autocorrelation"
    )

    ax.axhline(0, color="black", lw=0.8)

    confidence = 1.96 / np.sqrt(N)

    ax.axhline(
        confidence, color="red", ls="--", lw=0.8,
        label="95% confidence bounds"
    )
    ax.axhline(-confidence, color="red", ls="--", lw=0.8)

    for k in range(1, YEARS_TO_DISPLAY + 1):
        ax.axvline(k * YEAR, color="green", ls=":", lw=0.8)

    ax.set(
        xlabel="Lag (days)",
        ylabel="Autocorrelation",
        title=f"Autocorrelation of {var} - {label}"
    )

    ax.set_xlim(0, max_lag_display)
    ax.set_xticks([int(k * YEAR) for k in range(YEARS_TO_DISPLAY + 1)])
    ax.set_ylim(-1, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"figures/q5_{var}_autocorrelation.png", dpi=200)
    plt.close(fig)

    print(f"\n=== Autocorrelation of {var} ===")
    print(f"Number of samples: {N}")
    print(f"{'Lag (days)':>12} {'Autocorrelation':>18}")

    for k in range(0, MAX_LAG + 1, 365):
        print(f"{k:12d} {acf[k]:18.4f}")

print("\nQ5 completed.")


