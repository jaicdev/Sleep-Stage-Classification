import os
import mne
import numpy as np
import pywt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class SleepStageAnalysis:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.event_id = {
            'Sleep stage W': 1,
            'Sleep stage N1': 2,
            'Sleep stage N2': 3,
            'Sleep stage N3': 4,
            'Sleep stage R': 5
        }
        self.wavelet = 'db4'  # Daubechies 4 wavelet
        self.levels = 5  # Number of decomposition levels
        self.psg_files, self.hyp_files = self._get_files()

    def _get_files(self):
        psg_files, hyp_files = [], []
        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith('sleepscoring.edf'):
                hyp_files.append(os.path.join(self.folder_path, file))
            elif file.endswith('.edf'):
                psg_files.append(os.path.join(self.folder_path, file))
        return psg_files, hyp_files

    def remove_files(self, rem_list):
        self.psg_files = [f for f in self.psg_files if not any(r in f for r in rem_list)]
        self.hyp_files = [f for f in self.hyp_files if not any(r in f for r in rem_list)]

    def process_sleep_data(self, psg_file, hyp_file):
        raw = mne.io.read_raw_edf(psg_file, stim_channel='auto', preload=True)
        annot = mne.read_annotations(hyp_file)
        raw.set_annotations(annot, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=self.event_id, chunk_duration=30.)
        tmax = 30. - 1. / raw.info['sfreq']
        epochs = mne.Epochs(raw, events, self.event_id, tmin=0., tmax=tmax, baseline=None, preload=True)
        epochs.filter(l_freq=0.5, h_freq=None)  # High-pass filter at 0.5 Hz
        return epochs

    def wavelet_decomposition(self, signal):
        coeffs = pywt.wavedec(signal, wavelet=self.wavelet, level=self.levels)
        return coeffs

    def calculate_hjorth_params(self, signal):
        # Activity (variance of the signal)
        activity = np.var(signal)
        
        # Mobility
        diff_signal = np.diff(signal)
        mobility = np.sqrt(np.var(diff_signal) / activity)
        
        # Complexity
        diff_diff_signal = np.diff(diff_signal)
        complexity = np.sqrt(np.var(diff_diff_signal) / np.var(diff_signal)) / mobility
        
        return activity, mobility, complexity

    def extract_features(self, epochs):
        X = []
        for epoch in epochs:
            epoch_features = []
            for channel in epoch:
                # Perform wavelet decomposition
                coeffs = self.wavelet_decomposition(channel)
                
                # Calculate Hjorth parameters for each decomposition level
                for coeff in coeffs:
                    activity, mobility, complexity = self.calculate_hjorth_params(coeff)
                    epoch_features.extend([activity, mobility, complexity])
                
                # Add statistical features for each decomposition level
                for coeff in coeffs:
                    epoch_features.extend([
                        np.mean(coeff),
                        np.std(coeff),
                        stats.skew(coeff),
                        stats.kurtosis(coeff)
                    ])
            
            X.append(epoch_features)
        
        return np.array(X)

    def prepare_data(self, num_files=30):
        all_epochs = []
        for psg_file, hyp_file in zip(self.psg_files[:num_files], self.hyp_files[:num_files]):
            epochs = self.process_sleep_data(psg_file, hyp_file)
            all_epochs.append(epochs)
        combined_epochs = mne.concatenate_epochs(all_epochs)
        
        X = self.extract_features(combined_epochs)
        y = combined_epochs.events[:, 2]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self, X_train, y_train):
        models = {
            'Random Forest': make_pipeline(
                StandardScaler(),
                RandomForestClassifier(n_estimators=100, random_state=42)
            ),
            'Gradient Boosting': make_pipeline(
                StandardScaler(),
                GradientBoostingClassifier(n_estimators=100, random_state=42)
            ),
            'Support Vector Machine': make_pipeline(
                StandardScaler(),
                SVC(kernel='rbf', random_state=42)
            )
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
        
        return models

    def evaluate_models(self, models, X_test, y_test):
        results = {}
        for name, model in models.items():
            print(f"\nEvaluating {name}:")
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            print(f"Accuracy score: {acc:.2f}")
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=list(self.event_id.keys())))
        return results

    def plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.event_id.keys(), yticklabels=self.event_id.keys())
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def plot_model_comparison(self, results):
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values(), color='skyblue')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison')
        plt.show()

def main():
    # Initialize the analysis
    folder_path = 'E:/ME Disseratation/Data/hmc-sleep-staging/1.0.1/recordings/edf'
    analysis = SleepStageAnalysis(folder_path)

    # Remove specified files
    rem_list = ['SN010', 'SN011', 'SN012', 'SN013', 'SN014', 'SN017', 'SN018', 'SN026', 'SN030', 'SN032', 'SN033', 'SN135']
    analysis.remove_files(rem_list)

    # Prepare data
    X_train, X_test, y_train, y_test = analysis.prepare_data(num_files=30)

    # Train models
    models = analysis.train_models(X_train, y_train)

    # Evaluate models
    results = analysis.evaluate_models(models, X_test, y_test)

    # Plot results
    for name, model in models.items():
        y_pred = model.predict(X_test)
        analysis.plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {name}")

    analysis.plot_model_comparison(results)

if __name__ == "__main__":
    main()

