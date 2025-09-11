"""
Machine Learning Classifiers for Cyber-Attack Detection
Implements Random Forest, SVM, CNN, and LSTM classifiers
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import joblib
import logging
from typing import Tuple, Dict, Any, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestDetector:
    """
    Random Forest classifier for attack detection with optimized hyperparameters
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.training_time = 0
        self.prediction_time = 0
        
    def train(self, X: np.ndarray, y: np.ndarray, optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train the Random Forest classifier
        
        Args:
            X: Training features
            y: Training labels
            optimize_hyperparams: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary containing training metrics
        """
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize_hyperparams:
            # Hyperparameter optimization
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
            )
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_scaled, y)
        
        self.training_time = time.time() - start_time
        self.feature_importance = self.model.feature_importances_
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='f1_weighted')
        
        return {
            'training_time': self.training_time,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': self.feature_importance
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Make predictions on new data
        
        Args:
            X: Features for prediction
            
        Returns:
            Tuple of (predictions, probabilities, prediction_time)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        self.prediction_time = time.time() - start_time
        
        return predictions, probabilities, self.prediction_time
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions, probabilities, pred_time = self.predict(X)
        
        # Calculate metrics
        accuracy = (predictions == y).mean()
        report = classification_report(y, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y, predictions)
        
        # ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            auc_score = roc_auc_score(y, probabilities[:, 1])
        else:
            auc_score = roc_auc_score(y, probabilities, multi_class='ovr')
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'auc_score': auc_score,
            'prediction_time': pred_time,
            'predictions_per_second': len(X) / pred_time if pred_time > 0 else np.inf
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        logger.info(f"Model loaded from {filepath}")


class SVMDetector:
    """
    Support Vector Machine classifier with optimal kernel selection
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.training_time = 0
        self.prediction_time = 0
        self.best_kernel = None
        
    def train(self, X: np.ndarray, y: np.ndarray, optimize_kernel: bool = True) -> Dict[str, Any]:
        """
        Train the SVM classifier with optimal kernel selection
        
        Args:
            X: Training features
            y: Training labels
            optimize_kernel: Whether to optimize kernel selection
            
        Returns:
            Dictionary containing training metrics
        """
        start_time = time.time()
        
        # Scale features (crucial for SVM)
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize_kernel:
            # Test different kernels and hyperparameters
            param_grid = [
                {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 1]},
                {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4]},
                {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                {'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
            ]
            
            svm = SVC(random_state=self.random_state, probability=True)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
            )
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            self.best_kernel = grid_search.best_params_['kernel']
            
            logger.info(f"Best SVM parameters: {grid_search.best_params_}")
        else:
            self.model = SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True
            )
            self.model.fit(X_scaled, y)
            self.best_kernel = 'rbf'
        
        self.training_time = time.time() - start_time
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='f1_weighted')
        
        return {
            'training_time': self.training_time,
            'best_kernel': self.best_kernel,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'support_vectors': self.model.n_support_
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        self.prediction_time = time.time() - start_time
        
        return predictions, probabilities, self.prediction_time
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        predictions, probabilities, pred_time = self.predict(X)
        
        accuracy = (predictions == y).mean()
        report = classification_report(y, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y, predictions)
        
        if len(np.unique(y)) == 2:
            auc_score = roc_auc_score(y, probabilities[:, 1])
        else:
            auc_score = roc_auc_score(y, probabilities, multi_class='ovr')
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'auc_score': auc_score,
            'prediction_time': pred_time,
            'predictions_per_second': len(X) / pred_time if pred_time > 0 else np.inf
        }


class CNNDetector:
    """
    Convolutional Neural Network for time-series attack detection
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int = 2, random_state: int = 42):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random_state = random_state
        self.model = None
        self.scaler = MinMaxScaler()
        self.training_time = 0
        self.prediction_time = 0
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _build_model(self) -> tf.keras.Model:
        """Build CNN architecture for time-series analysis"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv1D(128, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv1D(256, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.5),
            
            # Dense Layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, 
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the CNN model
        
        Args:
            X: Training features (samples, timesteps, features)
            y: Training labels
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary containing training metrics
        """
        start_time = time.time()
        
        # Reshape and scale data
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Convert labels to categorical if multi-class
        if self.num_classes > 2:
            y_categorical = tf.keras.utils.to_categorical(y, self.num_classes)
        else:
            y_categorical = y
        
        # Build and compile model
        self.model = self._build_model()
        
        optimizer = optimizers.Adam(learning_rate=0.001)
        loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
        )
        
        # Train model
        history = self.model.fit(
            X_scaled, y_categorical,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        return {
            'training_time': self.training_time,
            'history': history.history,
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'epochs_trained': len(history.history['accuracy'])
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        
        # Reshape and scale data
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        probabilities = self.model.predict(X_scaled, verbose=0)
        
        if self.num_classes > 2:
            predictions = np.argmax(probabilities, axis=1)
        else:
            predictions = (probabilities > 0.5).astype(int).flatten()
            probabilities = np.column_stack([1 - probabilities, probabilities])
        
        self.prediction_time = time.time() - start_time
        
        return predictions, probabilities, self.prediction_time
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        predictions, probabilities, pred_time = self.predict(X)
        
        accuracy = (predictions == y).mean()
        report = classification_report(y, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y, predictions)
        
        if len(np.unique(y)) == 2:
            auc_score = roc_auc_score(y, probabilities[:, 1])
        else:
            auc_score = roc_auc_score(y, probabilities, multi_class='ovr')
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'auc_score': auc_score,
            'prediction_time': pred_time,
            'predictions_per_second': len(X) / pred_time if pred_time > 0 else np.inf
        }


class LSTMDetector:
    """
    LSTM Network for sequential pattern recognition in power system data
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int = 2, random_state: int = 42):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random_state = random_state
        self.model = None
        self.scaler = MinMaxScaler()
        self.training_time = 0
        self.prediction_time = 0
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
    def _build_model(self) -> tf.keras.Model:
        """Build LSTM architecture for sequential pattern recognition"""
        model = models.Sequential([
            # First LSTM Layer
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second LSTM Layer
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third LSTM Layer
            layers.LSTM(32, return_sequences=False),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Dense Layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Train the LSTM model"""
        start_time = time.time()
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Convert labels to categorical if multi-class
        if self.num_classes > 2:
            y_categorical = tf.keras.utils.to_categorical(y, self.num_classes)
        else:
            y_categorical = y
        
        # Build and compile model
        self.model = self._build_model()
        
        optimizer = optimizers.Adam(learning_rate=0.001)
        loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7
        )
        
        # Train model
        history = self.model.fit(
            X_scaled, y_categorical,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        return {
            'training_time': self.training_time,
            'history': history.history,
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'epochs_trained': len(history.history['accuracy'])
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        probabilities = self.model.predict(X_scaled, verbose=0)
        
        if self.num_classes > 2:
            predictions = np.argmax(probabilities, axis=1)
        else:
            predictions = (probabilities > 0.5).astype(int).flatten()
            probabilities = np.column_stack([1 - probabilities, probabilities])
        
        self.prediction_time = time.time() - start_time
        
        return predictions, probabilities, self.prediction_time
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        predictions, probabilities, pred_time = self.predict(X)
        
        accuracy = (predictions == y).mean()
        report = classification_report(y, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y, predictions)
        
        if len(np.unique(y)) == 2:
            auc_score = roc_auc_score(y, probabilities[:, 1])
        else:
            auc_score = roc_auc_score(y, probabilities, multi_class='ovr')
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'auc_score': auc_score,
            'prediction_time': pred_time,
            'predictions_per_second': len(X) / pred_time if pred_time > 0 else np.inf
        }