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
f0 = 1 / YEAR                    # frequency of the annual cycle found in Q2 (cycles/day)
M_LIMIT = YEAR / 2               # 182.6 days: keeping 1 value every M days, the annual cycle needs M < YEAR/2 (Shannon)

# Subsampling = keeping 1 value every M days: x[::M]
# new sampling frequency fs' = 1/M, new Nyquist frequency fN = 1/(2M).
# A component of frequency f > fN is not lost: it comes back at the aliased frequency
#     f_alias = | f - k/M |   with k the integer closest to f*M   (always between 0 and fN)


def spectrum(y, fs, pad=1):
    """Same amplitude spectrum as in q2_fourier.py: mean removed, rfft, A = 2|X|/n.
    pad > 1 adds zeros (zero-padding): the spectrum is interpolated, so the peaks are located more precisely."""
    n = len(y)
    Y = np.fft.rfft(y - y.mean(), n=pad * n)
    return np.fft.rfftfreq(pad * n, d=1 / fs), 2 * np.abs(Y) / n


def alias(f, fs):
    """Apparent frequency of a component of frequency f sampled at fs (theory)."""
    return np.abs(f - fs * np.round(f / fs))


# ------ 1- where does the annual cycle go? theory vs measurement -------

Ms = [7, 30, 90, 180, 240, 365, 380]   # subsampling steps studied (days)
rows = []
print("\n === Annual cycle after subsampling: theory vs measurement ===")
print(f"{'M (days)':>9} {'fN (c/yr)':>10} {'alias th. (c/yr)':>17} {'measured (c/yr)':>16} "
      f"{'apparent period (yr)':>21} {'amplitude (°C)':>15}")
for M in Ms:
    xs = x[::M]
    f_th = alias(f0, 1 / M)
    resolution = 1 / (len(xs) * M)              # frequency resolution of the subsampled series (cycles/day)
    if f_th < 2 * resolution:                   # M = 365: alias ~ 0, cannot be separated from the mean
        f_meas, a_meas = np.nan, np.nan
    else:
        fk, Ak = spectrum(xs, 1 / M, pad=8)
        i = np.argmax(Ak[1:]) + 1               # highest peak (f = 0 excluded)
        f_meas, a_meas = fk[i], Ak[i]
    rows.append((M, f_th, f_meas, a_meas))
    T_app = 1 / (f_meas * YEAR) if np.isfinite(f_meas) else 1 / (f_th * YEAR)
    print(f"{M:9d} {YEAR / (2 * M):10.3f} {f_th * YEAR:17.4f} {f_meas * YEAR:16.4f} {T_app:21.2f} {a_meas:15.2f}")
# expected: for M < 182.6 the peak stays at 1 cycle/year;
# M = 240 -> fake cycle of ~1.9 year; M = 365 -> alias ~ 0 (always the same day of the year, the cycle disappears);
# M = 380 -> fake cycle of ~26 years ("stroboscope" effect).
# The amplitude of the aliased peak stays ~ the amplitude of the annual cycle (~9 °C):
# aliasing moves the energy to a wrong frequency, it does not remove it.

# theoretical curve f_alias(M) for every step from 1 to 450 days, with the measured points on top
M_grid = np.arange(1, 451)
fig, ax = plt.subplots(figsize=(10, 5))
ax.axvspan(M_LIMIT, 450, color="r", alpha=0.07)
ax.plot(M_grid, np.minimum(YEAR / (2 * M_grid), 1.6), "k:", lw=1, label="Nyquist frequency fN = 1/(2M)")
ax.plot(M_grid, alias(f0, 1 / M_grid) * YEAR, lw=2, label="apparent frequency of the annual cycle (theory)")
for M, f_th, f_meas, a_meas in rows:
    if np.isfinite(f_meas):
        ax.plot(M, f_meas * YEAR, "o", ms=9, mfc="white", mec="r", mew=2)
        if M > M_LIMIT:
            ax.annotate(f"M = {M} d\nT = {1 / (f_meas * YEAR):.1f} yr", (M, f_meas * YEAR),
                        textcoords="offset points", xytext=(6, 8), fontsize=10, color="r")
ax.plot([], [], "o", ms=9, mfc="white", mec="r", mew=2, label="peak measured on the spectrum")
ax.axvline(M_LIMIT, color="r", ls="--", lw=0.8)
ax.text(M_LIMIT + 4, 1.5, "M = 182.6 d", color="r", fontsize=10, va="top")
ax.set_xlim(0, 450)
ax.set_ylim(0, 1.6)
ax.set(xlabel="Subsampling step M (days)", ylabel="Frequency (cycles/year)",
       title="Apparent frequency of the annual cycle after subsampling")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)
fig.tight_layout()
fig.savefig("figures/q3_alias_vs_step.png", dpi=200)
plt.close(fig)


# ------ 2- consequences on the signal and on its spectrum -------

# time domain: daily signal (gray) and subsampled points
cases = [(30, 6, "M = 30 days: f0 < fN, the annual cycle is well represented"),
         (180, None, "M = 180 days: f0 just below fN, ~2 points per year -> slow beating"),
         (240, 16, "M = 240 days: aliasing -> fake cycle of ~2 years"),
         (380, None, "M = 380 days: aliasing -> fake cycle of ~26 years")]
fig, ax = plt.subplots(len(cases), 1, figsize=(11, 11))
for a, (M, n_years, title) in zip(ax, cases):
    end = s.index[0] + pd.DateOffset(years=n_years) if n_years else s.index[-1]
    zoom = s.index <= end
    xs = s.iloc[::M]
    xs = xs[xs.index <= end]
    a.plot(s.index[zoom], x[zoom], lw=0.4, color="gray", label="daily signal")
    a.plot(xs.index, xs.values, "o-", ms=3, lw=1, label=f"1 value every {M} days")
    if M > M_LIMIT:
        # sine wave at the predicted aliased frequency fitted on the subsampled points (least squares)
        fa = alias(f0, 1 / M)
        t = (xs.index - s.index[0]).days.values
        B = np.column_stack([np.ones(len(t)), np.cos(2 * np.pi * fa * t), np.sin(2 * np.pi * fa * t)])
        c = np.linalg.lstsq(B, xs.values, rcond=None)[0]
        tt = np.arange(0, (end - s.index[0]).days)
        a.plot(s.index[0] + pd.to_timedelta(tt, unit="D"),
               c[0] + c[1] * np.cos(2 * np.pi * fa * tt) + c[2] * np.sin(2 * np.pi * fa * tt), "r", lw=2,
               label=f"apparent sine wave: T = {1 / (fa * YEAR):.1f} yr, amplitude {np.hypot(c[1], c[2]):.1f} °C")
    a.set_title(title, fontsize=11, loc="left")
    a.set_ylabel("°C")
    a.legend(loc="upper right", fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig("figures/q3_time_domain.png", dpi=200)
plt.close(fig)

# frequency domain: one spectrum per step, with the Nyquist frequency and the predicted position of the annual cycle
fig, ax = plt.subplots(2, 4, figsize=(15, 7))
ax = ax.ravel()
f, A = spectrum(x, fs)
ax[0].semilogy(f * YEAR, A, lw=0.5)
ax[0].axvline(1, color="r", ls="--", lw=0.8)
ax[0].set_xlim(0, 6)
ax[0].set_title("daily (M = 1)\nfN = 182.6 c/yr", fontsize=10)
for a, M in zip(ax[1:], Ms):
    fk, Ak = spectrum(x[::M], 1 / M, pad=4)
    fN = YEAR / (2 * M)
    a.semilogy(fk * YEAR, Ak, lw=0.8)
    a.axvline(fN, color="k", ls=":", lw=1.2)
    a.axvline(alias(f0, 1 / M) * YEAR, color="r", ls="--", lw=0.8)
    a.set_xlim(0, min(1.05 * fN, 6))
    a.set_title(f"M = {M} d ({'ok' if M < M_LIMIT else 'ALIASED'})\nfN = {fN:.2f} c/yr", fontsize=10)
for a in ax:
    a.set_xlabel("cycles/year")
ax[0].set_ylabel("Amplitude (°C)")
ax[4].set_ylabel("Amplitude (°C)")
fig.suptitle("Spectra after subsampling (black dotted: Nyquist, red dashed: predicted annual cycle)")
fig.tight_layout()
fig.savefig("figures/q3_spectra.png", dpi=200)
plt.close(fig)


# ------ 3- the weather is aliased too: anti-aliasing filter (M = 30 days) -------

# Even when the annual cycle is fine (M = 30 < 182.6), everything faster than 2M = 60 days
# (day-to-day weather) is folded into [0, fN] and raises the noise level of the spectrum.
# The fix is to low-pass filter BEFORE keeping 1 value every M days.
M = 30
nb = N // M                                           # number of complete blocks of 30 days
xc = x[:nb * M]
naive = xc[::M]                                       # 1 value every 30 days, no filter
block_mean = xc.reshape(nb, M).mean(axis=1)           # mean of each block of 30 days (moving average + subsampling)
filtered = signal.decimate(signal.decimate(xc, 5, ftype="fir", zero_phase=True),
                           6, ftype="fir", zero_phase=True)[:nb]   # FIR anti-aliasing filter, in 2 stages (5 x 6 = 30)

# energy check (Parseval): the variance above fN is not removed by the naive subsampling, it is folded back
X = np.fft.rfft(xc - xc.mean())
fx = np.fft.rfftfreq(len(xc), d=1 / fs)
P = np.abs(X[1:]) ** 2
share_hf = P[fx[1:] > 1 / (2 * M)].sum() / P.sum()    # share of the variance above the new Nyquist frequency


def noise_level(y):
    """median amplitude between 2.5 cycles/year and fN: the 'floor' of the spectrum, away from the annual peaks"""
    fk, Ak = spectrum(y, 1 / M)
    band = (fk * YEAR > 2.5) & (fk * YEAR < 0.95 * YEAR / (2 * M))
    return np.median(Ak[band])


gain_dB = 20 * np.log10(noise_level(naive) / noise_level(filtered))
print(f"\n === Anti-aliasing, M = {M} days ===")
print(f"variance of the daily signal              : {np.var(xc):6.2f} °C²")
print(f"share of the variance above fN            : {100 * share_hf:6.1f} %")
print(f"expected variance if this part is removed : {np.var(xc) * (1 - share_hf):6.2f} °C²")
print(f"variance, naive subsampling               : {np.var(naive):6.2f} °C²")
print(f"variance, block mean                      : {np.var(block_mean):6.2f} °C²")
print(f"variance, FIR anti-aliasing filter        : {np.var(filtered):6.2f} °C²")
print(f"noise level: naive vs filtered            : +{gain_dB:.1f} dB")
print(f"RMSE naive vs true monthly mean           : {np.sqrt(np.mean((naive - block_mean) ** 2)):6.2f} °C")
# expected: variance(naive) ~ variance(daily) (nothing lost, everything folded back),
# variance(filtered) ~ variance(daily) x (1 - share above fN), and a higher noise floor without filter.
# This is why climatologists publish monthly MEANS and not "the value of the 1st day of the month".

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
dates = s.index[:nb * M][::M]
zoom = dates <= dates[0] + pd.DateOffset(years=4)
zoom_d = s.index[:nb * M] <= dates[0] + pd.DateOffset(years=4)
ax[0].plot(s.index[:nb * M][zoom_d], xc[zoom_d], lw=0.4, color="gray", label="daily signal")
ax[0].plot(dates[zoom], naive[zoom], "o-", ms=3, lw=1, label="1 value / 30 days (naive)")
ax[0].plot(dates[zoom], filtered[zoom], "s-", ms=3, lw=1.5, label="anti-aliasing filter + subsampling")
ax[0].set(ylabel="Mean temperature (°C)", title="Time domain (first 4 years)")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
for y, lab in [(naive, "naive"), (block_mean, "block mean"), (filtered, "FIR anti-aliasing")]:
    fk, Ak = spectrum(y, 1 / M)
    ax[1].semilogy(fk * YEAR, Ak, lw=0.9, label=f"{lab} (var = {np.var(y):.1f} °C²)")
ax[1].axvline(YEAR / (2 * M), color="k", ls=":", lw=1.2)
ax[1].set(xlabel="cycles/year", ylabel="Amplitude (°C)",
          title=f"Spectra: noise floor +{gain_dB:.1f} dB without filter")
ax[1].legend(fontsize=9)
fig.suptitle(f"The weather faster than 60 days ({100 * share_hf:.1f} % of the variance) is aliased "
             f"if we do not filter before subsampling")
fig.tight_layout()
fig.savefig("figures/q3_antialiasing.png", dpi=200)
plt.close(fig)
