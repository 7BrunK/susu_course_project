import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import zscore
from scipy.signal import savgol_filter

DATASET_PATH = "mat"
OUTPUT_PATH = "results"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_cwru_file(filepath):
    mat = loadmat(filepath)
    keys = [k for k in mat.keys() if "DE_time" in k or "FE_time" in k or "BA_time" in k or "RPM" in k]
    return {key: mat[key].flatten() for key in keys}


def get_all_files(dataset_path):
    return [f for f in os.listdir(dataset_path) if f.endswith(".mat")]


def dataset_summary(files):
    summary = []

    for file in files:
        filepath = os.path.join(DATASET_PATH, file)
        data = load_cwru_file(filepath)

        row = {"file": file}
        for key, values in data.items():
            row[f"{key}_len"] = len(values)
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_PATH, "01_dataset_summary.csv"), index=False, sep=';')
    return summary_df


def visualize_signals(sample_data):
    plt.figure(figsize=(14, 8))

    for key, values in sample_data.items():
        if len(values) > 10:
            plt.plot(values[:5000], label=key)

    plt.title("Original Signals")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "02_signals.png"))
    plt.close()


def statistical_analysis(sample_data):
    stats = []

    for key, values in sample_data.items():
        if len(values) > 10:
            stats.append({
                "signal": key,
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
                "skew": pd.Series(values).skew(),
                "kurtosis": pd.Series(values).kurtosis()
            })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(OUTPUT_PATH, "03_statistics.csv"), index=False, sep=';')
    return stats_df


def analyze_missing_outliers(sample_data):
    missing_outliers = []

    valid_signals = {
        key: values for key, values in sample_data.items()
        if len(values) > 10
    }

    fig, axes = plt.subplots(len(valid_signals), 1, figsize=(10, 4 * len(valid_signals)))

    if len(valid_signals) == 1:
        axes = [axes]

    for idx, (key, values) in enumerate(valid_signals.items()):
        missing = np.isnan(values).sum()
        z_scores = np.abs(zscore(values))
        outliers = np.sum(z_scores > 3)

        missing_outliers.append({
            "signal": key,
            "missing_values": missing,
            "outliers_count": outliers
        })

        sns.boxplot(x=values[:5000], ax=axes[idx])
        axes[idx].set_title(f"Boxplot {key}")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "04_boxplots.png"))
    plt.close()

    missing_df = pd.DataFrame(missing_outliers)
    missing_df.to_csv(os.path.join(OUTPUT_PATH, "04_missing_outliers.csv"), index=False, sep=';')

    return missing_df


def analyze_ranges(sample_data):
    ranges = []

    valid_signals = {
        key: values for key, values in sample_data.items()
        if len(values) > 10
    }

    fig, axes = plt.subplots(len(valid_signals), 1, figsize=(10, 4 * len(valid_signals)))

    if len(valid_signals) == 1:
        axes = [axes]

    for idx, (key, values) in enumerate(valid_signals.items()):
        ranges.append({
            "signal": key,
            "min": np.min(values),
            "max": np.max(values),
            "range": np.max(values) - np.min(values)
        })

        axes[idx].hist(values[:5000], bins=50)
        axes[idx].set_title(f"Histogram {key}")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "05_histograms.png"))
    plt.close()

    ranges_df = pd.DataFrame(ranges)
    ranges_df.to_csv(os.path.join(OUTPUT_PATH, "05_ranges.csv"), index=False, sep=';')

    return ranges_df


def correlation_analysis(sample_data):
    corr_data = {}

    valid_lengths = [len(values[:5000]) for values in sample_data.values() if len(values) > 10]
    min_len = min(valid_lengths)

    for key, values in sample_data.items():
        if len(values) > 10:
            corr_data[key] = values[:min_len]

    corr_df = pd.DataFrame(corr_data)
    corr_matrix = corr_df.corr()

    corr_matrix.to_csv(os.path.join(OUTPUT_PATH, "06_correlation.csv"), sep=';')

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "06_corr_heatmap.png"))
    plt.close()

    return corr_matrix


def noise_analysis(sample_data):
    noise_stats = []

    plt.figure(figsize=(14, 8))

    for key, values in sample_data.items():
        if len(values) <= 10:
            continue

        signal = values[:5000]

        window_length = min(101, len(signal))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length <= 3:
            continue

        smooth = savgol_filter(signal, window_length, 3)
        noise = signal - smooth

        noise_stats.append({
            "signal": key,
            "noise_mean": np.mean(noise),
            "noise_std": np.std(noise)
        })

        plt.plot(noise, label=f"{key} noise")

    plt.legend()
    plt.title("Noise Signals")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "07_noise.png"))
    plt.close()

    noise_df = pd.DataFrame(noise_stats)
    noise_df.to_csv(os.path.join(OUTPUT_PATH, "07_noise_stats.csv"), index=False, sep=';')
    return noise_df


def main():
    files = get_all_files(DATASET_PATH)

    dataset_summary(files)

    sample_file = os.path.join(DATASET_PATH, files[0])
    sample_data = load_cwru_file(sample_file)

    visualize_signals(sample_data)
    statistical_analysis(sample_data)
    analyze_missing_outliers(sample_data)
    analyze_ranges(sample_data)
    correlation_analysis(sample_data)
    noise_analysis(sample_data)

    print(f"Все результаты сохранены в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()