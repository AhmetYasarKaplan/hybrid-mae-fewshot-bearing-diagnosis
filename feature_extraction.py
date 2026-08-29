import numpy as np
from scipy import stats
from scipy.signal import hilbert, welch


def extract_time_features(x):
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    skewness = stats.skew(x)
    kurtosis = stats.kurtosis(x)

    crest = peak / (rms + 1e-10)
    shape = rms / (np.mean(np.abs(x)) + 1e-10)
    impulse = peak / (np.mean(np.abs(x)) + 1e-10)
    clearance = peak / (np.mean(np.sqrt(np.abs(x))) + 1e-10)**2
    margin = peak / (np.mean(np.abs(x))**2 + 1e-10)
    energy = np.sum(x**2)
    zcr = np.sum(np.abs(np.diff(np.sign(x)))) / (2 * n)
    p2p = np.max(x) - np.min(x)

    return [mean, std, rms, peak, skewness, kurtosis,
            crest, shape, impulse, clearance, margin,
            energy, zcr, p2p]


def extract_freq_features(x, fs=48000):
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))

    total_power = np.sum(psd)
    psd_norm = psd / (total_power + 1e-10)
    mean_freq = np.sum(freqs * psd_norm)
    rms_freq = np.sqrt(np.sum(freqs**2 * psd_norm))
    freq_std = np.sqrt(np.sum((freqs - mean_freq)**2 * psd_norm))
    freq_skew = np.sum((freqs - mean_freq)**3 * psd_norm) / (freq_std**3 + 1e-10)
    freq_kurt = np.sum((freqs - mean_freq)**4 * psd_norm) / (freq_std**4 + 1e-10)

    n_bands = 5
    band_size = len(psd) // n_bands
    band_energies = []
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else len(psd)
        band_energies.append(np.sum(psd[start:end]) / (total_power + 1e-10))

    peak_freq = freqs[np.argmax(psd)]

    return [total_power, mean_freq, rms_freq, freq_std,
            freq_skew, freq_kurt, peak_freq] + band_energies


def extract_envelope_spectrum_features(x, fs=48000):
    x = np.asarray(x)
    x = x - np.mean(x)
    envelope = np.abs(hilbert(x))
    envelope = envelope - np.mean(envelope)
    freqs, psd = welch(envelope, fs=fs, nperseg=min(256, len(envelope)))

    total_power = np.sum(psd)
    psd_norm = psd / (total_power + 1e-10)
    peak_freq = freqs[np.argmax(psd)]
    centroid = np.sum(freqs * psd_norm)
    rms_freq = np.sqrt(np.sum(freqs**2 * psd_norm))

    n_bands = 5
    band_size = len(psd) // n_bands
    band_energies = []
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else len(psd)
        band_energies.append(np.sum(psd[start:end]) / (total_power + 1e-10))

    return [total_power, peak_freq, centroid, rms_freq] + band_energies


def extract_features(x, fs=48000, feature_set="standard"):
    if feature_set not in {"standard", "envelope", "all"}:
        raise ValueError("feature_set must be one of: standard, envelope, all")

    features = []
    if feature_set in {"standard", "all"}:
        features.extend(extract_time_features(x))
        features.extend(extract_freq_features(x, fs))
    if feature_set in {"envelope", "all"}:
        features.extend(extract_envelope_spectrum_features(x, fs))
    return np.array(features)


def extract_features_batch(X, fs=48000, feature_set="standard"):
    if hasattr(X, 'numpy'):
        X = X.numpy()

    if X.ndim == 3 and X.shape[1] > 1:
        features = []
        for i in range(len(X)):
            per_channel = []
            for ch in range(X.shape[1]):
                per_channel.append(extract_features(X[i, ch], fs, feature_set))
            features.append(np.concatenate(per_channel))
        return np.array(features)

    if X.ndim == 3:
        X = X.squeeze(1)

    features = []
    for i in range(len(X)):
        features.append(extract_features(X[i], fs, feature_set))
    return np.array(features)


STANDARD_FEATURE_NAMES = [
    "mean", "std", "rms", "peak", "skewness", "kurtosis",
    "crest_factor", "shape_factor", "impulse_factor",
    "clearance_factor", "margin_factor", "energy", "zcr", "p2p",
    "total_power", "mean_freq", "rms_freq", "freq_std",
    "freq_skew", "freq_kurt", "peak_freq",
    "band_energy_1", "band_energy_2", "band_energy_3",
    "band_energy_4", "band_energy_5",
]

ENVELOPE_FEATURE_NAMES = [
    "env_total_power", "env_peak_freq", "env_centroid", "env_rms_freq",
    "env_band_energy_1", "env_band_energy_2", "env_band_energy_3",
    "env_band_energy_4", "env_band_energy_5",
]


def get_feature_names(feature_set="standard"):
    if feature_set == "standard":
        return STANDARD_FEATURE_NAMES
    if feature_set == "envelope":
        return ENVELOPE_FEATURE_NAMES
    if feature_set == "all":
        return STANDARD_FEATURE_NAMES + ENVELOPE_FEATURE_NAMES
    raise ValueError("feature_set must be one of: standard, envelope, all")
