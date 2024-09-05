import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

class SleepStageEDA:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.event_id = {
            'Sleep stage W': 1,
            'Sleep stage N1': 2,
            'Sleep stage N2': 3,
            'Sleep stage N3': 4,
            'Sleep stage R': 5
        }
        self.psg_files, self.hyp_files = self._get_files()

    def _get_files(self):
        psg_files, hyp_files = [], []
        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith('sleepscoring.edf'):
                hyp_files.append(os.path.join(self.folder_path, file))
            elif file.endswith('.edf'):
                psg_files.append(os.path.join(self.folder_path, file))
        return psg_files, hyp_files

    def load_data(self, psg_file, hyp_file):
        raw = mne.io.read_raw_edf(psg_file, stim_channel='auto', preload=True)
        annot = mne.read_annotations(hyp_file)
        raw.set_annotations(annot, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=self.event_id, chunk_duration=30.)
        return raw, events

    def plot_signal(self, raw, start=0, duration=60):
        """Plots raw EEG signals over the specified time duration"""
        raw.plot(start=start, duration=duration, n_channels=10, scalings='auto', title="EEG Signal")

    def plot_power_spectral_density(self, raw):
        """Plots the Power Spectral Density (PSD) of the EEG signals"""
        plt.figure(figsize=(10, 6))
        raw.plot_psd(fmax=50, show=True)
        plt.title("Power Spectral Density (PSD)")
        plt.show()

    def plot_event_distribution(self, events):
        """Plots the distribution of sleep stage events"""
        event_counts = Counter(events[:, 2])
        stage_names = [self.event_id_to_name(e) for e in event_counts.keys()]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=stage_names, y=list(event_counts.values()), palette='Blues_d')
        plt.title("Sleep Stage Event Distribution")
        plt.xlabel("Sleep Stages")
        plt.ylabel("Number of Events")
        plt.tight_layout()
        plt.show()

    def event_id_to_name(self, event_id):
        return list(self.event_id.keys())[list(self.event_id.values()).index(event_id)]

    def plot_epoch_distribution(self, epochs):
        """Plots distribution of epoch features (e.g., variance, mean) across channels"""
        variances = [np.var(epoch) for epoch in epochs.get_data()]
        means = [np.mean(epoch) for epoch in epochs.get_data()]

        # Plot variance
        plt.figure(figsize=(10, 6))
        plt.hist(variances, bins=50, color='skyblue', alpha=0.7)
        plt.title("Variance Distribution across Epochs")
        plt.xlabel("Variance")
        plt.ylabel("Frequency")
        plt.show()

        # Plot mean
        plt.figure(figsize=(10, 6))
        plt.hist(means, bins=50, color='orange', alpha=0.7)
        plt.title("Mean Distribution across Epochs")
        plt.xlabel("Mean")
        plt.ylabel("Frequency")
        plt.show()

    def plot_correlation_matrix(self, raw):
        """Plots correlation matrix between EEG channels"""
        data = raw.get_data()
        correlation_matrix = np.corrcoef(data)

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation Matrix between EEG Channels")
        plt.show()

def main():
    # Initialize the EDA
    folder_path = 'E:/ME Disseratation/Data/hmc-sleep-staging/1.0.1/recordings/edf'
    eda = SleepStageEDA(folder_path)

    # Example: Select a file for EDA
    psg_file = eda.psg_files[0]
    hyp_file = eda.hyp_files[0]

    # Load data
    raw, events = eda.load_data(psg_file, hyp_file)

    # Plot raw EEG signals
    eda.plot_signal(raw, start=0, duration=60)

    # Plot Power Spectral Density (PSD)
    eda.plot_power_spectral_density(raw)

    # Plot event distribution (sleep stages)
    eda.plot_event_distribution(events)

    # Process epochs
    tmax = 30. - 1. / raw.info['sfreq']
    epochs = mne.Epochs(raw, events, eda.event_id, tmin=0., tmax=tmax, baseline=None, preload=True)

    # Plot epoch variance and mean distributions
    eda.plot_epoch_distribution(epochs)

    # Plot correlation matrix between channels
    eda.plot_correlation_matrix(raw)

if __name__ == "__main__":
    main()

