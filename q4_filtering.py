import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from data import load_series

plt.rcParams.update({'font.size': 12, "figure.dpi": 100})

# ------ 0- the signal -------

s = load_series("tavg")          # daily mean temperature (°C), same series as in q2_fourier.py
x = s.values                     # numpy array
N = len(x)                       # number of samples
fs = 1.0                         # sampling frequency: 1 sample per day -> Nyquist = 0.5 cycle/day
YEAR = 365.25                    # length of a year in days
ZOOM_YEAR = 2003                 # year shown in the zoom (2003 heat wave)

# Instead of a single filter we use a bank of 2 low-pass filters, so that the signal is split
# into 3 parts whose sum gives back exactly the original signal:
#     x = T (trend, periods > 5 years) + S (seasons, 30 days to 5 years) + R (fast weather, periods < 30 days)
#     T = LP_5years(x),   S = LP_30days(x) - T,   R = x - LP_30days(x)
fc_fast = 1 / 30                 # cut-off between seasons and weather (cycles/day)
fc_trend = 1 / (5 * YEAR)        # cut-off between trend and seasons (cycles/day)
order = 4                        # Butterworth order: flat pass band, so the annual cycle amplitude is not modified


def spectrum(y, fs):
    """Same amplitude spectrum as in q2_fourier.py: mean removed, rfft, A = 2|X|/n."""
    Y = np.fft.rfft(y - y.mean())
    return np.fft.rfftfreq(len(y), d=1 / fs), 2 * np.abs(Y) / len(y)


def lowpass(y, fc):
    """Butterworth low-pass applied forward and backward (sosfiltfilt): zero phase, no delay.
    - 'sos' form (second-order sections): the (b, a) form is numerically unstable for a cut-off as low as 1/(5 years)
    - edges: the signal is extended by an EVEN reflection (mirror) over ~3 cut-off periods. The default odd
      reflection forces the output through the value of the first/last day (a winter day here),
      which makes the trend dive at both ends."""
    sos = signal.butter(order, fc, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, y, padtype="even", padlen=int(min(len(y) - 1, 3 / fc)))


# ------ 1- choice of the filter: frequency responses and delays -------

fr = np.logspace(np.log10(0.05 / YEAR), np.log10(0.49), 3000)   # frequencies from 1/20 years to Nyquist (cycles/day)
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for n in (2, 4, 8):
    sos = signal.butter(n, fc_fast, fs=fs, output="sos")
    _, h = signal.sosfreqz(sos, worN=fr, fs=fs)
    ax[0].plot(fr * YEAR, 20 * np.log10(np.abs(h) ** 2 + 1e-12), lw=2 if n == order else 1,
               label=f"Butterworth order {n}, forward-backward (|H|²)" + ("  <- used" if n == order else ""))
    b, a = signal.butter(n, fc_fast, fs=fs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, gd = signal.group_delay((b, a), w=fr, fs=fs)
    ax[1].plot(fr * YEAR, gd, lw=1.2, label=f"Butterworth {n}, causal")
L_ma = 30                                                        # moving average of 30 days
h_ma = np.abs(np.sin(np.pi * fr * L_ma) / (L_ma * np.sin(np.pi * fr)))   # its response is a periodic sinc
ax[0].plot(fr * YEAR, 20 * np.log10(h_ma + 1e-12), lw=1, label="moving average 30 days (side lobes)")
h_fir = signal.firwin(241, fc_fast, fs=fs, window="hamming")    # FIR filter, Hamming window, 241 coefficients
_, hf = signal.freqz(h_fir, worN=fr, fs=fs)
ax[0].plot(fr * YEAR, 20 * np.log10(np.abs(hf) + 1e-12), lw=1, label="FIR Hamming, 241 coefficients")
ax[1].axhline((L_ma - 1) / 2, color="C3", lw=1, label="moving average, causal (14.5 d)")
ax[1].axhline(120, color="C4", lw=1, label="FIR, causal (120 d)")
ax[1].axhline(0, color="k", ls="--", lw=1.5, label="forward-backward: 0 d")
for a in ax:
    a.axvline(1, color="r", ls="--", lw=0.8)                     # annual cycle
    a.axvline(fc_fast * YEAR, color="k", ls=":", lw=1)           # cut-off
    a.axvspan(YEAR / 10, YEAR / 3, color="orange", alpha=0.1)    # synoptic weather (3-10 days)
ax[0].text(YEAR / 5.5, -8, "weather\n3-10 days", ha="center", va="top", fontsize=10)
ax[0].set(ylim=(-100, 5), ylabel="Gain (dB)", title="Low-pass filters with a 30-day cut-off")
ax[0].legend(loc="lower left", fontsize=9)
ax[1].set(xscale="log", ylim=(-5, 140), ylabel="Group delay (days)", xlabel="Frequency (cycles/year, log scale)")
ax[1].legend(fontsize=8, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig("figures/q4_filter_responses.png", dpi=200)
plt.close(fig)
# expected: the moving average lets through some fast weather (side lobes at -13 dB);
# the FIR is very selective but delays the signal by 120 days if it is causal;
# Butterworth 4 is a good compromise, and forward-backward filtering removes the delay.
# Note: a moving average of 365 days has zeros exactly at 1, 2, 3... cycles/year: it removes the annual cycle
# and its harmonics, which is why it is used for the trend (compared with T below).


# ------ 2- splitting the signal: x = T + S + R -------

slow = lowpass(x, fc_fast)       # everything slower than 30 days (T + S)
T = lowpass(x, fc_trend)         # trend
S = slow - T                     # seasons
R = x - slow                     # fast weather
print("\n === Filter bank: x = T + S + R ===")
print(f"max reconstruction error  : {np.max(np.abs(x - (T + S + R))):.1e} °C")
for name, c in (("trend T", T), ("seasons S", S), ("weather R", R)):
    print(f"{name:10s}: {100 * np.var(c) / np.var(x):5.1f} % of the variance, std = {c.std():.2f} °C")
# the 3 shares do not add up exactly to 100 %: the filters are not ideal, the bands overlap a little
# near the cut-offs, so S and R are slightly correlated (cross terms)

# amplitude of the annual cycle inside S (least squares fit of a sine wave of period 1 year), to compare with Q2
t = np.arange(N)
B = np.column_stack([np.cos(2 * np.pi * t / YEAR), np.sin(2 * np.pi * t / YEAR)])
c = np.linalg.lstsq(B, S - S.mean(), rcond=None)[0]
print(f"amplitude of the annual cycle in S: {np.hypot(*c):.2f} °C (to compare with the annual peak of Q2)")

# warming trend, computed in 3 ways
edge = int(2 * YEAR)                                             # 2 years removed at each end (edge effects)
years = (s.index - s.index[0]).days.values / YEAR
slope_T = 10 * np.polyfit(years[edge:-edge], T[edge:-edge], 1)[0]
annual = s.groupby(s.index.year).agg(["mean", "count"])
annual = annual[annual["count"] >= 360]["mean"]                 # complete years only
p_annual = np.polyfit(annual.index, annual.values, 1)
recent = annual[annual.index >= 1980]
slope_1980 = 10 * np.polyfit(recent.index, recent.values, 1)[0]
print(f"trend from T              : {slope_T:+.2f} °C/decade")
print(f"trend from annual means   : {10 * p_annual[0]:+.2f} °C/decade "
      f"(total {p_annual[0] * (annual.index[-1] - annual.index[0]):+.1f} °C over {annual.index[0]}-{annual.index[-1]})")
print(f"trend since 1980          : {slope_1980:+.2f} °C/decade")

ma365 = s.rolling(365, center=True).mean()
mid_year = pd.to_datetime(annual.index.astype(str) + "-07-01")
fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
ax[0].plot(s.index, x, lw=0.3, color="gray", label="daily signal x")
ax[0].plot(s.index, slow, lw=0.6, label="low-pass 30 days (T + S)")
ax[0].plot(s.index, T, "r", lw=2, label="trend T")
ax[0].set_title("Raw signal and slow component", fontsize=11, loc="left")
ax[1].plot(s.index, ma365, lw=1, color="orange", label="moving average 365 days")
ax[1].plot(s.index, T, "r", lw=2, label=f"Butterworth, periods > 5 years: {slope_T:+.2f} °C/decade")
ax[1].plot(mid_year, annual.values, "ko", ms=3, alpha=0.6,
           label=f"annual means: {10 * p_annual[0]:+.2f} °C/decade (since 1980: {slope_1980:+.2f})")
ax[1].set_title("T: climate trend (gray zones: edge effects)", fontsize=11, loc="left")
for a in ax[:2]:
    a.axvspan(s.index[0], s.index[edge], color="k", alpha=0.07, lw=0)
    a.axvspan(s.index[-edge], s.index[-1], color="k", alpha=0.07, lw=0)
ax[2].plot(s.index, S, lw=0.6)
ax[2].set_title(f"S: seasonal cycle ({100 * np.var(S) / np.var(x):.0f} % of the variance)", fontsize=11, loc="left")
ax[3].plot(s.index, R, lw=0.25, color="C2")
ax[3].set_title(f"R: fast weather, periods < 30 days ({100 * np.var(R) / np.var(x):.0f} % of the variance)",
                fontsize=11, loc="left")
for a in ax:
    a.set_ylabel("°C")
ax[0].legend(loc="upper left", fontsize=9, ncol=3)
ax[1].legend(loc="upper left", fontsize=9, ncol=3)
fig.tight_layout()
fig.savefig("figures/q4_decomposition.png", dpi=200)
plt.close(fig)


# ------ 3- before / after filtering in the frequency domain -------

fig, ax = plt.subplots(figsize=(10, 5))
for c, lab, col in [(x, "raw signal", "gray"), (S, "S: seasons", "C0"), (R, "R: weather", "C2"), (T, "T: trend", "r")]:
    f, A = spectrum(c, fs)
    ax.loglog(f[1:] * YEAR, A[1:], lw=0.8, color=col, label=lab)
ax.axvline(fc_fast * YEAR, color="k", ls=":", lw=1)
ax.axvline(fc_trend * YEAR, color="k", ls=":", lw=1)
ax.set_ylim(1e-4, None)
ax.set(xlabel="Frequency (cycles/year)", ylabel="Amplitude (°C)",
       title="Amplitude spectra of the 3 components (dotted: cut-offs 1/5 years and 1/30 days)")
ax.legend(loc="lower left", fontsize=10)
fig.tight_layout()
fig.savefig("figures/q4_spectra_components.png", dpi=200)
plt.close(fig)
# expected: each component keeps only its band; the annual peak (and its harmonic at 2 cycles/year) is entirely in S


# ------ 4- zoom on one year: causal filter vs zero phase -------

sos = signal.butter(order, fc_fast, fs=fs, output="sos")
causal = signal.sosfilt(sos, x - x[0]) + x[0]                    # forward only (like a real-time filter)
# delay measured by cross-correlation between the causal and the zero-phase outputs
m = 400                                                           # start/end removed (filter transient)
a_, b_ = causal[m:-m] - causal[m:-m].mean(), slow[m:-m] - slow[m:-m].mean()
lags = np.arange(-60, 61)
cc = [np.dot(a_[60 + k:len(a_) - 60 + k], b_[60:len(b_) - 60]) for k in lags]
delay_meas = lags[np.argmax(cc)]
b, a = signal.butter(order, fc_fast, fs=fs)
delay_th = signal.group_delay((b, a), w=[1 / YEAR], fs=fs)[1][0]  # theoretical group delay at 1 cycle/year
print(f"\n === Causal filter delay: measured {delay_meas} days, theory {delay_th:.1f} days ===")

zoom = s.index.year == ZOOM_YEAR
summer = zoom & s.index.month.isin([6, 7, 8])
i_max = np.flatnonzero(summer)[np.argmax(R[summer])]              # largest summer anomaly
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(s.index[zoom], x[zoom], lw=0.8, color="gray", label="daily signal")
ax[0].plot(s.index[zoom], slow[zoom], lw=2, label="Butterworth forward-backward (zero phase)")
ax[0].plot(s.index[zoom], causal[zoom], "r--", lw=1.5,
           label=f"Butterworth causal: delay {delay_meas} d (theory {delay_th:.1f} d)")
ax[0].plot(s.index[zoom], s.rolling(30, center=True).mean().values[zoom], lw=1.2, label="moving average 30 d")
ax[0].set(ylabel="°C", title=f"{ZOOM_YEAR}: slow component, effect of the phase of the filter")
ax[0].legend(loc="upper left", fontsize=9)
r = R[zoom]
ax[1].fill_between(s.index[zoom], 0, r, where=r > 0, color="r", alpha=0.5, lw=0)
ax[1].fill_between(s.index[zoom], 0, r, where=r < 0, color="C0", alpha=0.5, lw=0)
ax[1].axhline(2 * R.std(), color="k", ls=":", lw=0.8)
ax[1].axhline(-2 * R.std(), color="k", ls=":", lw=0.8)
ax[1].annotate(f"summer max {R[i_max]:+.1f} °C ({s.index[i_max]:%d/%m})", (s.index[i_max], R[i_max]),
               textcoords="offset points", xytext=(12, 10), fontsize=10)
ax[1].set(ylabel="°C", title="R: weather anomalies (dotted: ±2 std)")
fig.tight_layout()
fig.savefig("figures/q4_zoom_causal_vs_zero_phase.png", dpi=200)
plt.close(fig)
# note: a heat wave of ~2 weeks is close to the 30-day cut-off, so part of it goes into the slow component


# ------ 5- interpretation: when is the weather most variable? -------

Rs = pd.Series(R, index=s.index)
std_month = Rs.groupby(s.index.month).std()
winter = Rs[s.index.month.isin([12, 1, 2])].std()
summer_std = Rs[s.index.month.isin([6, 7, 8])].std()
std_decade = Rs.groupby((s.index.year // 10) * 10).std()
print(f"\n === Day-to-day variability (std of R): winter {winter:.2f} °C, summer {summer_std:.2f} °C, "
      f"ratio {winter / summer_std:.2f} ===")

fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.4, 1]})
colors = ["C0" if k in (12, 1, 2) else ("r" if k in (6, 7, 8) else "gray") for k in range(1, 13)]
ax[0].bar(range(1, 13), std_month.values, color=colors)
ax[0].set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax[0].set(ylabel="std of R (°C)", title=f"Std of the fast weather by month (winter / summer = {winter / summer_std:.2f})")
ax[1].plot(std_decade.index, std_decade.values, "o-")
ax[1].set(xlabel="decade", ylabel="std of R (°C)", title="... and by decade")
fig.tight_layout()
fig.savefig("figures/q4_weather_variability.png", dpi=200)
plt.close(fig)
# expected: the weather is more variable in winter (stronger north-south temperature contrast,
# more active low-pressure systems, alternation of mild oceanic and cold continental air) than in summer
