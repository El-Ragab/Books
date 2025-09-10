"""
Statistical Methods for Power System Cyber-Attack Detection
Implements Pearson correlation, Z-score anomaly detection, transient fault analysis, and statistical baselines
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, welch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional
import logging
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PearsonCorrelationAnalyzer:
    """
    Pearson correlation analysis for multi-IED coordination and attack detection
    """
    
    def __init__(self, correlation_threshold: float = 0.8):
        self.correlation_threshold = correlation_threshold
        self.baseline_correlations = None
        self.feature_names = None
        self.training_time = 0
        self.prediction_time = 0
        
    def train(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Establish baseline correlation patterns from normal operation data
        
        Args:
            X: Training data (normal operation)
            feature_names: Names of features for interpretation
            
        Returns:
            Dictionary containing baseline correlation analysis
        """
        start_time = time.time()
        
        # Convert to DataFrame for easier handling
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        self.feature_names = feature_names
        df = pd.DataFrame(X, columns=feature_names)
        
        # Calculate baseline correlation matrix
        self.baseline_correlations = df.corr(method='pearson')
        
        # Identify highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr_value = self.baseline_correlations.iloc[i, j]
                if abs(corr_value) >= self.correlation_threshold:
                    high_corr_pairs.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'correlation': corr_value
                    })
        
        # Calculate statistical properties
        corr_values = self.baseline_correlations.values
        upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
        
        self.training_time = time.time() - start_time
        
        return {
            'training_time': self.training_time,
            'baseline_correlations': self.baseline_correlations,
            'high_correlation_pairs': high_corr_pairs,
            'mean_correlation': upper_triangle.mean(),
            'std_correlation': upper_triangle.std(),
            'max_correlation': upper_triangle.max(),
            'min_correlation': upper_triangle.min(),
            'num_high_corr_pairs': len(high_corr_pairs)
        }
    
    def detect_anomalies(self, X: np.ndarray, window_size: int = 100, 
                        threshold_multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detect anomalies based on correlation pattern changes
        
        Args:
            X: Test data
            window_size: Size of sliding window for correlation calculation
            threshold_multiplier: Multiplier for anomaly threshold
            
        Returns:
            Tuple of (anomaly_labels, correlation_deviations, prediction_time)
        """
        if self.baseline_correlations is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        
        df = pd.DataFrame(X, columns=self.feature_names)
        anomaly_labels = np.zeros(len(X))
        correlation_deviations = np.zeros(len(X))
        
        # Sliding window correlation analysis
        for i in range(window_size, len(X)):
            # Calculate correlation for current window
            window_data = df.iloc[i-window_size:i]
            current_corr = window_data.corr(method='pearson')
            
            # Calculate deviation from baseline
            corr_diff = current_corr - self.baseline_correlations
            deviation = np.sqrt(np.mean(corr_diff.values**2))  # RMSE deviation
            correlation_deviations[i] = deviation
            
            # Determine if anomalous
            if deviation > threshold_multiplier * correlation_deviations[:i].std():
                anomaly_labels[i] = 1
        
        self.prediction_time = time.time() - start_time
        
        return anomaly_labels, correlation_deviations, self.prediction_time
    
    def analyze_correlation_changes(self, X: np.ndarray, window_size: int = 100) -> Dict[str, Any]:
        """
        Detailed analysis of correlation changes over time
        
        Args:
            X: Data to analyze
            window_size: Size of sliding window
            
        Returns:
            Dictionary containing detailed correlation analysis
        """
        df = pd.DataFrame(X, columns=self.feature_names)
        correlation_timeline = []
        
        for i in range(window_size, len(X), window_size//2):  # 50% overlap
            window_data = df.iloc[i-window_size:i]
            current_corr = window_data.corr(method='pearson')
            
            # Calculate metrics for this window
            corr_values = current_corr.values
            upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
            
            correlation_timeline.append({
                'window_start': i-window_size,
                'window_end': i,
                'mean_correlation': upper_triangle.mean(),
                'std_correlation': upper_triangle.std(),
                'max_correlation': upper_triangle.max(),
                'min_correlation': upper_triangle.min()
            })
        
        return {
            'correlation_timeline': correlation_timeline,
            'baseline_mean': self.baseline_correlations.values[
                np.triu_indices_from(self.baseline_correlations.values, k=1)
            ].mean()
        }
    
    def visualize_correlation_matrix(self, save_path: Optional[str] = None):
        """Visualize baseline correlation matrix"""
        if self.baseline_correlations is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(self.baseline_correlations, dtype=bool))
        
        sns.heatmap(
            self.baseline_correlations,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title('Baseline Correlation Matrix (Normal Operation)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ZScoreAnomalyDetector:
    """
    Z-score based anomaly detection with adaptive thresholds
    """
    
    def __init__(self, threshold: float = 3.0, adaptive: bool = True):
        self.threshold = threshold
        self.adaptive = adaptive
        self.baseline_stats = None
        self.scaler = StandardScaler()
        self.training_time = 0
        self.prediction_time = 0
        
    def train(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Establish baseline statistics from normal operation data
        
        Args:
            X: Training data (normal operation)
            
        Returns:
            Dictionary containing baseline statistics
        """
        start_time = time.time()
        
        # Calculate baseline statistics
        self.baseline_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'median': np.median(X, axis=0),
            'q25': np.percentile(X, 25, axis=0),
            'q75': np.percentile(X, 75, axis=0),
            'iqr': np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        }
        
        # Fit scaler for standardization
        self.scaler.fit(X)
        
        self.training_time = time.time() - start_time
        
        return {
            'training_time': self.training_time,
            'baseline_stats': self.baseline_stats,
            'feature_means': self.baseline_stats['mean'],
            'feature_stds': self.baseline_stats['std']
        }
    
    def detect_anomalies(self, X: np.ndarray, window_size: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detect anomalies using Z-score analysis
        
        Args:
            X: Test data
            window_size: Window size for adaptive threshold calculation
            
        Returns:
            Tuple of (anomaly_labels, z_scores, prediction_time)
        """
        if self.baseline_stats is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        
        # Calculate Z-scores
        z_scores = np.abs((X - self.baseline_stats['mean']) / self.baseline_stats['std'])
        max_z_scores = np.max(z_scores, axis=1)  # Maximum Z-score per sample
        
        anomaly_labels = np.zeros(len(X))
        
        if self.adaptive:
            # Adaptive threshold based on sliding window
            for i in range(window_size, len(X)):
                window_z_scores = max_z_scores[i-window_size:i]
                adaptive_threshold = np.mean(window_z_scores) + self.threshold * np.std(window_z_scores)
                
                if max_z_scores[i] > adaptive_threshold:
                    anomaly_labels[i] = 1
        else:
            # Fixed threshold
            anomaly_labels = (max_z_scores > self.threshold).astype(int)
        
        self.prediction_time = time.time() - start_time
        
        return anomaly_labels, max_z_scores, self.prediction_time
    
    def analyze_feature_anomalies(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Analyze which features contribute most to anomalies
        
        Args:
            X: Test data
            
        Returns:
            Dictionary containing feature-wise anomaly analysis
        """
        z_scores = np.abs((X - self.baseline_stats['mean']) / self.baseline_stats['std'])
        
        feature_anomaly_rates = (z_scores > self.threshold).mean(axis=0)
        feature_max_z_scores = np.max(z_scores, axis=0)
        feature_mean_z_scores = np.mean(z_scores, axis=0)
        
        return {
            'feature_anomaly_rates': feature_anomaly_rates,
            'feature_max_z_scores': feature_max_z_scores,
            'feature_mean_z_scores': feature_mean_z_scores,
            'most_anomalous_features': np.argsort(feature_anomaly_rates)[::-1]
        }


class TransientFaultAnalyzer:
    """
    Transient fault signature analysis for attack differentiation
    """
    
    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate
        self.fault_signatures = None
        self.attack_signatures = None
        self.training_time = 0
        self.prediction_time = 0
        
    def train(self, normal_data: np.ndarray, fault_data: np.ndarray, 
              attack_data: np.ndarray) -> Dict[str, Any]:
        """
        Learn signatures of normal operation, faults, and attacks
        
        Args:
            normal_data: Normal operation data
            fault_data: Transient fault data
            attack_data: Cyber-attack data
            
        Returns:
            Dictionary containing learned signatures
        """
        start_time = time.time()
        
        # Extract signatures for each type
        self.normal_signatures = self._extract_signatures(normal_data)
        self.fault_signatures = self._extract_signatures(fault_data)
        self.attack_signatures = self._extract_signatures(attack_data)
        
        self.training_time = time.time() - start_time
        
        return {
            'training_time': self.training_time,
            'normal_signatures': self.normal_signatures,
            'fault_signatures': self.fault_signatures,
            'attack_signatures': self.attack_signatures
        }
    
    def _extract_signatures(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Extract characteristic signatures from data
        
        Args:
            data: Input data
            
        Returns:
            Dictionary containing extracted signatures
        """
        signatures = {}
        
        # Time domain features
        signatures['mean'] = np.mean(data, axis=0)
        signatures['std'] = np.std(data, axis=0)
        signatures['skewness'] = stats.skew(data, axis=0)
        signatures['kurtosis'] = stats.kurtosis(data, axis=0)
        signatures['peak_to_peak'] = np.ptp(data, axis=0)
        
        # Frequency domain features
        freq_features = []
        for i in range(data.shape[1]):
            freqs, psd = welch(data[:, i], fs=self.sampling_rate, nperseg=min(256, len(data)//4))
            
            # Dominant frequency
            dominant_freq = freqs[np.argmax(psd)]
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            
            # Spectral rolloff (95% of energy)
            cumsum_psd = np.cumsum(psd)
            rolloff_idx = np.argmax(cumsum_psd >= 0.95 * cumsum_psd[-1])
            spectral_rolloff = freqs[rolloff_idx]
            
            freq_features.append({
                'dominant_frequency': dominant_freq,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'total_power': np.sum(psd)
            })
        
        signatures['frequency_features'] = freq_features
        
        # Transient characteristics
        transient_features = []
        for i in range(data.shape[1]):
            signal = data[:, i]
            
            # Find peaks
            peaks, _ = find_peaks(np.abs(signal), height=np.std(signal))
            
            # Rate of change
            rate_of_change = np.diff(signal)
            
            transient_features.append({
                'num_peaks': len(peaks),
                'max_rate_of_change': np.max(np.abs(rate_of_change)),
                'mean_rate_of_change': np.mean(np.abs(rate_of_change)),
                'settling_time': self._calculate_settling_time(signal)
            })
        
        signatures['transient_features'] = transient_features
        
        return signatures
    
    def _calculate_settling_time(self, signal: np.ndarray, tolerance: float = 0.02) -> float:
        """
        Calculate settling time of a signal
        
        Args:
            signal: Input signal
            tolerance: Settling tolerance (2% by default)
            
        Returns:
            Settling time in samples
        """
        final_value = signal[-1]
        tolerance_band = tolerance * abs(final_value)
        
        # Find last time signal exits tolerance band
        outside_tolerance = np.abs(signal - final_value) > tolerance_band
        
        if not np.any(outside_tolerance):
            return 0
        
        last_exit = np.where(outside_tolerance)[0][-1]
        settling_time = len(signal) - last_exit
        
        return settling_time / self.sampling_rate  # Convert to seconds
    
    def classify_event(self, data: np.ndarray) -> Tuple[str, Dict[str, float], float]:
        """
        Classify an event as normal, fault, or attack based on signatures
        
        Args:
            data: Event data to classify
            
        Returns:
            Tuple of (classification, confidence_scores, prediction_time)
        """
        if self.fault_signatures is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        
        # Extract signatures from test data
        test_signatures = self._extract_signatures(data)
        
        # Calculate similarity scores
        normal_score = self._calculate_signature_similarity(test_signatures, self.normal_signatures)
        fault_score = self._calculate_signature_similarity(test_signatures, self.fault_signatures)
        attack_score = self._calculate_signature_similarity(test_signatures, self.attack_signatures)
        
        # Normalize scores
        total_score = normal_score + fault_score + attack_score
        confidence_scores = {
            'normal': normal_score / total_score,
            'fault': fault_score / total_score,
            'attack': attack_score / total_score
        }
        
        # Classification
        classification = max(confidence_scores, key=confidence_scores.get)
        
        self.prediction_time = time.time() - start_time
        
        return classification, confidence_scores, self.prediction_time
    
    def _calculate_signature_similarity(self, sig1: Dict[str, Any], sig2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two signatures
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Similarity score
        """
        similarity_scores = []
        
        # Time domain similarity
        for feature in ['mean', 'std', 'skewness', 'kurtosis']:
            if feature in sig1 and feature in sig2:
                # Normalized correlation
                corr = np.corrcoef(sig1[feature], sig2[feature])[0, 1]
                if not np.isnan(corr):
                    similarity_scores.append(abs(corr))
        
        # Frequency domain similarity
        if 'frequency_features' in sig1 and 'frequency_features' in sig2:
            freq_similarities = []
            for f1, f2 in zip(sig1['frequency_features'], sig2['frequency_features']):
                for feature in ['dominant_frequency', 'spectral_centroid', 'spectral_rolloff']:
                    if feature in f1 and feature in f2:
                        # Normalized difference
                        diff = abs(f1[feature] - f2[feature]) / max(abs(f1[feature]), abs(f2[feature]), 1e-10)
                        freq_similarities.append(1 - diff)
            
            if freq_similarities:
                similarity_scores.append(np.mean(freq_similarities))
        
        return np.mean(similarity_scores) if similarity_scores else 0.0


class StatisticalBaseline:
    """
    Statistical baseline establishment for normal system behavior
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.baseline_stats = None
        self.control_limits = None
        self.training_time = 0
        
    def establish_baseline(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Establish statistical baseline from normal operation data
        
        Args:
            X: Normal operation data
            feature_names: Names of features
            
        Returns:
            Dictionary containing baseline statistics
        """
        start_time = time.time()
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Calculate comprehensive statistics
        self.baseline_stats = {}
        self.control_limits = {}
        
        alpha = 1 - self.confidence_level
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            
            # Basic statistics
            mean = np.mean(feature_data)
            std = np.std(feature_data, ddof=1)
            median = np.median(feature_data)
            
            # Distribution parameters
            skew = stats.skew(feature_data)
            kurt = stats.kurtosis(feature_data)
            
            # Confidence intervals
            ci_lower, ci_upper = stats.t.interval(
                self.confidence_level, 
                len(feature_data) - 1,
                loc=mean, 
                scale=stats.sem(feature_data)
            )
            
            # Control limits (3-sigma)
            ucl = mean + 3 * std  # Upper control limit
            lcl = mean - 3 * std  # Lower control limit
            
            # Percentiles
            percentiles = np.percentile(feature_data, [1, 5, 25, 50, 75, 95, 99])
            
            # Normality test
            _, p_value_shapiro = stats.shapiro(feature_data[:min(5000, len(feature_data))])
            is_normal = p_value_shapiro > 0.05
            
            self.baseline_stats[feature_name] = {
                'mean': mean,
                'std': std,
                'median': median,
                'skewness': skew,
                'kurtosis': kurt,
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'range': np.ptp(feature_data),
                'iqr': percentiles[5] - percentiles[2],  # Q3 - Q1
                'percentiles': {
                    'p1': percentiles[0], 'p5': percentiles[1], 'p25': percentiles[2],
                    'p50': percentiles[3], 'p75': percentiles[4], 'p95': percentiles[5],
                    'p99': percentiles[6]
                },
                'confidence_interval': (ci_lower, ci_upper),
                'is_normal': is_normal,
                'shapiro_p_value': p_value_shapiro
            }
            
            self.control_limits[feature_name] = {
                'upper_control_limit': ucl,
                'lower_control_limit': lcl,
                'center_line': mean
            }
        
        self.training_time = time.time() - start_time
        
        return {
            'training_time': self.training_time,
            'baseline_stats': self.baseline_stats,
            'control_limits': self.control_limits,
            'confidence_level': self.confidence_level
        }
    
    def detect_statistical_anomalies(self, X: np.ndarray, 
                                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect statistical anomalies based on established baseline
        
        Args:
            X: Test data
            feature_names: Names of features
            
        Returns:
            Dictionary containing anomaly detection results
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline not established yet. Call establish_baseline() first.")
        
        if feature_names is None:
            feature_names = list(self.baseline_stats.keys())
        
        anomaly_results = {}
        
        for i, feature_name in enumerate(feature_names):
            if feature_name not in self.baseline_stats:
                continue
                
            feature_data = X[:, i]
            baseline = self.baseline_stats[feature_name]
            control_limit = self.control_limits[feature_name]
            
            # Control chart violations
            ucl_violations = feature_data > control_limit['upper_control_limit']
            lcl_violations = feature_data < control_limit['lower_control_limit']
            control_violations = ucl_violations | lcl_violations
            
            # Statistical tests
            # One-sample t-test
            t_stat, t_p_value = stats.ttest_1samp(feature_data, baseline['mean'])
            
            # Kolmogorov-Smirnov test (if we had reference distribution)
            # For now, use percentile-based detection
            percentile_anomalies = (
                (feature_data < baseline['percentiles']['p1']) |
                (feature_data > baseline['percentiles']['p99'])
            )
            
            anomaly_results[feature_name] = {
                'control_violations': control_violations,
                'ucl_violations': ucl_violations,
                'lcl_violations': lcl_violations,
                'percentile_anomalies': percentile_anomalies,
                'violation_rate': np.mean(control_violations),
                'percentile_anomaly_rate': np.mean(percentile_anomalies),
                't_test_p_value': t_p_value,
                'mean_shift': np.mean(feature_data) - baseline['mean'],
                'std_change': np.std(feature_data) / baseline['std']
            }
        
        return anomaly_results
    
    def generate_control_chart(self, X: np.ndarray, feature_name: str, 
                             save_path: Optional[str] = None):
        """
        Generate control chart for a specific feature
        
        Args:
            X: Data to plot
            feature_name: Name of feature to plot
            save_path: Path to save the chart
        """
        if feature_name not in self.baseline_stats:
            raise ValueError(f"Feature {feature_name} not found in baseline statistics")
        
        feature_idx = list(self.baseline_stats.keys()).index(feature_name)
        feature_data = X[:, feature_idx]
        
        control_limit = self.control_limits[feature_name]
        
        plt.figure(figsize=(12, 6))
        
        # Plot data
        plt.plot(feature_data, 'b-', linewidth=1, alpha=0.7, label='Data')
        
        # Plot control limits
        plt.axhline(y=control_limit['center_line'], color='g', linestyle='-', 
                   linewidth=2, label='Center Line')
        plt.axhline(y=control_limit['upper_control_limit'], color='r', linestyle='--', 
                   linewidth=2, label='Upper Control Limit')
        plt.axhline(y=control_limit['lower_control_limit'], color='r', linestyle='--', 
                   linewidth=2, label='Lower Control Limit')
        
        # Highlight violations
        ucl_violations = feature_data > control_limit['upper_control_limit']
        lcl_violations = feature_data < control_limit['lower_control_limit']
        
        if np.any(ucl_violations):
            plt.scatter(np.where(ucl_violations)[0], feature_data[ucl_violations], 
                       color='red', s=50, zorder=5, label='UCL Violations')
        
        if np.any(lcl_violations):
            plt.scatter(np.where(lcl_violations)[0], feature_data[lcl_violations], 
                       color='red', s=50, zorder=5, label='LCL Violations')
        
        plt.title(f'Control Chart for {feature_name}')
        plt.xlabel('Sample Number')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()