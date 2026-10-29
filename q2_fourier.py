import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from data import load_series

plt.rcParams.update({'font.size': 12, "figure.dpi": 100})

# ------ 0- the signal -------

s = load_series("tavg")          # daily mean temperature (°C), 27394 values, no missing day
x = s.values                     # numpy array for the FFT
N = len(x)                       # number of samples
fs = 1.0                         # sampling frequency: 1 sample per day -> Nyquist = 0.5 cycle/day
YEAR = 365.25                    # length of a year in days


# ------ 1- FFT and amplitude spectrum -------

mean = x.mean()                  # we remove the mean first, otherwise the f = 0 bin is huge and hides everything
X = np.fft.rfft(x - mean)        # rfft: FFT of a real signal, keeps only the positive frequencies (0 to Nyquist)
f = np.fft.rfftfreq(N, d=1/fs)   # frequency of each bin in cycles/day, step = 1/N (resolution of 1/75 cycle per year)
A = 2 * np.abs(X) / N            # amplitude spectrum in °C: a sine wave of amplitude a gives a peak of height a
period = np.full_like(f, np.inf)
period[1:] = 1 / f[1:]           # period in days (infinite for f = 0)

# spectrum vs frequency: full range (log scale) and zoom on the low frequencies
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].semilogy(f[1:], A[1:], lw=0.5)   # log scale because the annual peak is ~100x larger than the rest
ax[0].set(xlabel="Frequency (cycles/day)", ylabel="Amplitude (°C)",
          title="Amplitude spectrum of tavg (full range up to Nyquist = 0.5 cycle/day)")
ax[1].plot(f, A, lw=0.8)
for k in (1, 2, 3):              # annual frequency and its first harmonics
    ax[1].axvline(k / YEAR, color="r", ls="--", lw=0.8)
ax[1].set_xlim(0, 0.012)         # 0 to ~4.4 cycles/year
ax[1].set(xlabel="Frequency (cycles/day)", ylabel="Amplitude (°C)",
          title="Zoom on low frequencies (red dashed lines: 1/365, 2/365, 3/365)")
fig.tight_layout()
fig.savefig("figures/q2_spectrum_freq.png", dpi=200)
plt.close(fig)

# spectrum vs period, which is easier to read: the annual peak sits at 365 days
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogx(period[1:], A[1:], lw=0.8)
ax.axvline(YEAR, color="r", ls="--", lw=0.8, label="365 days")
ax.set_xlim(2, 2 * N)            # from the shortest visible period (2 days, Nyquist) to the record length
ax.set(xlabel="Period (days, log scale)", ylabel="Amplitude (°C)",
       title="Amplitude spectrum of tavg vs period")
ax.legend()
fig.tight_layout()
fig.savefig("figures/q2_spectrum_period.png", dpi=200)
plt.close(fig)


# ------ 2- identify the peaks -------

# find_peaks keeps local maxima; prominence = how much a peak stands out from its neighbours (in °C)
peaks, _ = find_peaks(A, prominence=0.15)
peaks = peaks[np.argsort(A[peaks])[::-1]][:8]   # the 8 largest, sorted by amplitude
power = A[1:] ** 2 / 2                           # power of each sine wave = amplitude² / 2
print("\n === Main peaks of the spectrum of tavg ===")
print(f"{'freq (cycles/day)':>18} {'period (days)':>14} {'cycles/year':>12} {'amplitude (°C)':>15} {'% of variance':>14}")
for p in peaks:
    print(f"{f[p]:18.5f} {period[p]:14.1f} {f[p] * YEAR:12.2f} {A[p]:15.2f} {100 * A[p]**2 / 2 / power.sum():14.1f}")
# expected: a huge peak at 365 days (~9.2 °C, about 75 % of the variance),
# a small harmonic at ~183 days (2 cycles/year, the annual cycle is not a perfect sine wave),
# peaks at very long periods (thousands of days) that come from the warming trend
# (a trend is not periodic, so the FFT spreads it over the lowest frequencies),
# and small peaks around 100 days that are only weather noise (not multiples of 1 cycle/year)


# ------ 3- reconstruction with the K largest components -------

# we keep the mean + the K bins with the largest amplitude, set all the others to 0, and go back to time with the inverse FFT
order = np.argsort(np.abs(X[1:]))[::-1] + 1     # bins sorted by decreasing amplitude (bin 0 = the mean, handled apart)

def reconstruct(K):
    Y = np.zeros_like(X)
    Y[order[:K]] = X[order[:K]]                 # keep only the K largest sine waves
    return np.fft.irfft(Y, n=N) + mean          # back to the time domain, then add the mean again

Ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
rmse = []
print("\n === Reconstruction error vs K ===")
for K in Ks:
    r = reconstruct(K)
    e = np.sqrt(np.mean((x - r) ** 2))          # RMSE: typical error in °C between the real and the rebuilt signal
    rmse.append(e)
    expl = 100 * (1 - np.var(x - r) / np.var(x))   # share of the variance explained by the K components
    print(f"K = {K:5d} | RMSE = {e:.2f} °C | variance explained = {expl:.1f} %")

# signal vs reconstructions, zoom on 3 years so that the curves are readable
zoom = (s.index >= "2018-01-01") & (s.index <= "2020-12-31")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(s.index[zoom], x[zoom], lw=0.6, color="gray", label="real signal")
for K in (1, 3, 5):
    ax.plot(s.index[zoom], reconstruct(K)[zoom], lw=1.5,
            label=f"K = {K} (RMSE {rmse[Ks.index(K)]:.2f} °C)")
ax.set(xlabel="Date", ylabel="Mean temperature (°C)", title="Reconstruction of tavg with the K largest components")
ax.legend(loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig("figures/q2_reconstruction.png", dpi=200)
plt.close(fig)

# error as a function of K
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(Ks, rmse, "o-")
ax.axhline(x.std(), color="k", ls=":", label=f"std of the signal ({x.std():.2f} °C, K = 0)")
ax.set(xlabel="Number of components K (log scale)", ylabel="RMSE (°C)",
       title="Reconstruction error vs K")
ax.legend()
fig.tight_layout()
fig.savefig("figures/q2_error_vs_K.png", dpi=200)
plt.close(fig)
# expected: the error falls from 7.46 °C (K = 0) to ~3.7 °C with K = 1 (the annual cycle alone),
# then decreases very slowly: what is left is the day-to-day weather, spread over thousands of
# frequencies with no dominant one, so many components are needed to rebuild it.
