"""
Unsupervised Learning Techniques for Anomaly Detection
Implements K-means, PCA, and Isolation Forest for cyber-attack detection
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import Tuple, Dict, Any, Optional, List
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMeansAnomalyDetector:
    """
    K-means clustering based anomaly detector for power system data
    """
    
    def __init__(self, n_clusters: int = 2, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.anomaly_threshold = None
        self.training_time = 0
        self.prediction_time = 0
        
    def train(self, X: np.ndarray, optimize_clusters: bool = True) -> Dict[str, Any]:
        """
        Train K-means clustering model
        
        Args:
            X: Training data (normal operation data)
            optimize_clusters: Whether to optimize number of clusters
            
        Returns:
            Dictionary containing training metrics
        """
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize_clusters:
            # Elbow method and silhouette analysis
            inertias = []
            silhouette_scores = []
            k_range = range(2, min(11, len(X) // 10))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            # Select optimal k based on silhouette score
            optimal_k = k_range[np.argmax(silhouette_scores)]
            self.n_clusters = optimal_k
            
            logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Train final model
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.model.fit(X_scaled)
        
        # Calculate cluster centers and anomaly threshold
        self.cluster_centers = self.model.cluster_centers_
        
        # Calculate distances to nearest cluster center for threshold setting
        distances = self._calculate_distances(X_scaled)
        self.anomaly_threshold = np.percentile(distances, 95)  # 95th percentile as threshold
        
        self.training_time = time.time() - start_time
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X_scaled, self.model.labels_)
        inertia = self.model.inertia_
        
        return {
            'training_time': self.training_time,
            'n_clusters': self.n_clusters,
            'silhouette_score': silhouette_avg,
            'inertia': inertia,
            'anomaly_threshold': self.anomaly_threshold,
            'cluster_centers': self.cluster_centers
        }
    
    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate minimum distance to cluster centers"""
        distances = []
        for point in X:
            min_distance = min([np.linalg.norm(point - center) 
                               for center in self.cluster_centers])
            distances.append(min_distance)
        return np.array(distances)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict anomalies in new data
        
        Args:
            X: Test data
            
        Returns:
            Tuple of (anomaly_labels, distances, prediction_time)
            anomaly_labels: 1 for anomaly, 0 for normal
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        X_scaled = self.scaler.transform(X)
        
        # Calculate distances to nearest cluster center
        distances = self._calculate_distances(X_scaled)
        
        # Classify as anomaly if distance exceeds threshold
        anomaly_labels = (distances > self.anomaly_threshold).astype(int)
        
        self.prediction_time = time.time() - start_time
        
        return anomaly_labels, distances, self.prediction_time
    
    def evaluate(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate anomaly detection performance
        
        Args:
            X: Test data
            y_true: True labels (if available)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        anomaly_labels, distances, pred_time = self.predict(X)
        
        results = {
            'prediction_time': pred_time,
            'anomaly_rate': anomaly_labels.mean(),
            'mean_distance': distances.mean(),
            'std_distance': distances.std(),
            'predictions_per_second': len(X) / pred_time if pred_time > 0 else np.inf
        }
        
        if y_true is not None:
            # Calculate supervised metrics
            accuracy = (anomaly_labels == y_true).mean()
            tn = ((anomaly_labels == 0) & (y_true == 0)).sum()
            fp = ((anomaly_labels == 1) & (y_true == 0)).sum()
            fn = ((anomaly_labels == 0) & (y_true == 1)).sum()
            tp = ((anomaly_labels == 1) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            })
        
        return results
    
    def visualize_clusters(self, X: np.ndarray, save_path: Optional[str] = None):
        """Visualize clusters using PCA for dimensionality reduction"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Use PCA for visualization if data is high-dimensional
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            centers_pca = pca.transform(self.cluster_centers)
        else:
            X_pca = X_scaled
            centers_pca = self.cluster_centers
        
        # Create plot
        plt.figure(figsize=(10, 8))
        labels = self.model.predict(X_scaled)
        
        # Plot data points
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        
        # Plot cluster centers
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.title('K-means Clustering Results')
        plt.xlabel('First Principal Component' if X_scaled.shape[1] > 2 else 'Feature 1')
        plt.ylabel('Second Principal Component' if X_scaled.shape[1] > 2 else 'Feature 2')
        plt.colorbar(scatter)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class PCADetector:
    """
    Principal Component Analysis based anomaly detector
    """
    
    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = 0.95):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.reconstruction_threshold = None
        self.training_time = 0
        self.prediction_time = 0
        
    def train(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Train PCA model for anomaly detection
        
        Args:
            X: Training data (normal operation data)
            
        Returns:
            Dictionary containing training metrics
        """
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of components if not specified
        if self.n_components is None:
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum_var >= self.variance_threshold) + 1
        
        # Train final PCA model
        self.model = PCA(n_components=self.n_components)
        X_transformed = self.model.fit_transform(X_scaled)
        
        # Calculate reconstruction errors for threshold setting
        X_reconstructed = self.model.inverse_transform(X_transformed)
        reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Set threshold as 95th percentile of reconstruction errors
        self.reconstruction_threshold = np.percentile(reconstruction_errors, 95)
        
        self.training_time = time.time() - start_time
        
        return {
            'training_time': self.training_time,
            'n_components': self.n_components,
            'explained_variance_ratio': self.model.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.model.explained_variance_ratio_),
            'reconstruction_threshold': self.reconstruction_threshold
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict anomalies based on reconstruction error
        
        Args:
            X: Test data
            
        Returns:
            Tuple of (anomaly_labels, reconstruction_errors, prediction_time)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        X_scaled = self.scaler.transform(X)
        
        # Transform and reconstruct
        X_transformed = self.model.transform(X_scaled)
        X_reconstructed = self.model.inverse_transform(X_transformed)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Classify as anomaly if reconstruction error exceeds threshold
        anomaly_labels = (reconstruction_errors > self.reconstruction_threshold).astype(int)
        
        self.prediction_time = time.time() - start_time
        
        return anomaly_labels, reconstruction_errors, self.prediction_time
    
    def evaluate(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate anomaly detection performance"""
        anomaly_labels, reconstruction_errors, pred_time = self.predict(X)
        
        results = {
            'prediction_time': pred_time,
            'anomaly_rate': anomaly_labels.mean(),
            'mean_reconstruction_error': reconstruction_errors.mean(),
            'std_reconstruction_error': reconstruction_errors.std(),
            'predictions_per_second': len(X) / pred_time if pred_time > 0 else np.inf
        }
        
        if y_true is not None:
            # Calculate supervised metrics
            accuracy = (anomaly_labels == y_true).mean()
            tn = ((anomaly_labels == 0) & (y_true == 0)).sum()
            fp = ((anomaly_labels == 1) & (y_true == 0)).sum()
            fn = ((anomaly_labels == 0) & (y_true == 1)).sum()
            tp = ((anomaly_labels == 1) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            })
        
        return results


class IsolationForestDetector:
    """
    Isolation Forest based anomaly detector
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.training_time = 0
        self.prediction_time = 0
        
    def train(self, X: np.ndarray, optimize_contamination: bool = True) -> Dict[str, Any]:
        """
        Train Isolation Forest model
        
        Args:
            X: Training data (normal operation data)
            optimize_contamination: Whether to optimize contamination parameter
            
        Returns:
            Dictionary containing training metrics
        """
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize_contamination:
            # Try different contamination values
            contamination_values = [0.01, 0.05, 0.1, 0.15, 0.2]
            best_score = -np.inf
            best_contamination = self.contamination
            
            for cont in contamination_values:
                model = IsolationForest(
                    contamination=cont,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                anomaly_scores = model.fit(X_scaled).decision_function(X_scaled)
                # Use negative mean anomaly score as optimization metric
                score = -np.mean(anomaly_scores)
                
                if score > best_score:
                    best_score = score
                    best_contamination = cont
            
            self.contamination = best_contamination
            logger.info(f"Optimal contamination: {best_contamination}")
        
        # Train final model
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        
        self.training_time = time.time() - start_time
        
        # Calculate training metrics
        anomaly_scores = self.model.decision_function(X_scaled)
        
        return {
            'training_time': self.training_time,
            'contamination': self.contamination,
            'mean_anomaly_score': anomaly_scores.mean(),
            'std_anomaly_score': anomaly_scores.std(),
            'min_anomaly_score': anomaly_scores.min(),
            'max_anomaly_score': anomaly_scores.max()
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict anomalies using Isolation Forest
        
        Args:
            X: Test data
            
        Returns:
            Tuple of (anomaly_labels, anomaly_scores, prediction_time)
            anomaly_labels: 1 for anomaly, 0 for normal
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and anomaly scores
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to (1 for anomaly, 0 for normal)
        anomaly_labels = (predictions == -1).astype(int)
        
        self.prediction_time = time.time() - start_time
        
        return anomaly_labels, anomaly_scores, self.prediction_time
    
    def evaluate(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate anomaly detection performance"""
        anomaly_labels, anomaly_scores, pred_time = self.predict(X)
        
        results = {
            'prediction_time': pred_time,
            'anomaly_rate': anomaly_labels.mean(),
            'mean_anomaly_score': anomaly_scores.mean(),
            'std_anomaly_score': anomaly_scores.std(),
            'predictions_per_second': len(X) / pred_time if pred_time > 0 else np.inf
        }
        
        if y_true is not None:
            # Calculate supervised metrics
            accuracy = (anomaly_labels == y_true).mean()
            tn = ((anomaly_labels == 0) & (y_true == 0)).sum()
            fp = ((anomaly_labels == 1) & (y_true == 0)).sum()
            fn = ((anomaly_labels == 0) & (y_true == 1)).sum()
            tp = ((anomaly_labels == 1) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            })
        
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.contamination = model_data['contamination']
        logger.info(f"Model loaded from {filepath}")