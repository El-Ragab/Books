"""
Jacobian Matrix Analysis for Power System Stability Monitoring
Implements Jacobian calculation and analysis for cyber-attack detection
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals, cond, det
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional
import logging
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JacobianAnalyzer:
    """
    Jacobian matrix analysis for power system stability and attack detection
    """
    
    def __init__(self):
        self.baseline_jacobian = None
        self.baseline_eigenvalues = None
        self.baseline_condition_number = None
        self.stability_threshold = None
        self.training_time = 0
        self.prediction_time = 0
        
    def calculate_jacobian(self, bus_voltages: np.ndarray, bus_angles: np.ndarray,
                          y_matrix: np.ndarray, p_injections: np.ndarray,
                          q_injections: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian matrix for power flow equations
        
        Args:
            bus_voltages: Bus voltage magnitudes
            bus_angles: Bus voltage angles (in radians)
            y_matrix: System admittance matrix
            p_injections: Active power injections
            q_injections: Reactive power injections
            
        Returns:
            Jacobian matrix
        """
        n_buses = len(bus_voltages)
        
        # Initialize Jacobian submatrices
        J11 = np.zeros((n_buses, n_buses))  # ∂P/∂δ
        J12 = np.zeros((n_buses, n_buses))  # ∂P/∂V
        J21 = np.zeros((n_buses, n_buses))  # ∂Q/∂δ  
        J22 = np.zeros((n_buses, n_buses))  # ∂Q/∂V
        
        # Extract conductance and susceptance matrices
        G = np.real(y_matrix)
        B = np.imag(y_matrix)
        
        for i in range(n_buses):
            Vi = bus_voltages[i]
            δi = bus_angles[i]
            
            for j in range(n_buses):
                Vj = bus_voltages[j]
                δj = bus_angles[j]
                
                Gij = G[i, j]
                Bij = B[i, j]
                
                δij = δi - δj
                
                if i == j:
                    # Diagonal elements
                    # J11[i,i] = ∂Pi/∂δi
                    J11[i, i] = -q_injections[i] - Vi**2 * B[i, i]
                    
                    # J12[i,i] = ∂Pi/∂Vi
                    J12[i, i] = p_injections[i] / Vi + Vi * G[i, i]
                    
                    # J21[i,i] = ∂Qi/∂δi  
                    J21[i, i] = p_injections[i] - Vi**2 * G[i, i]
                    
                    # J22[i,i] = ∂Qi/∂Vi
                    J22[i, i] = q_injections[i] / Vi - Vi * B[i, i]
                    
                else:
                    # Off-diagonal elements
                    # J11[i,j] = ∂Pi/∂δj
                    J11[i, j] = Vi * Vj * (Gij * np.sin(δij) - Bij * np.cos(δij))
                    
                    # J12[i,j] = ∂Pi/∂Vj
                    J12[i, j] = Vi * (Gij * np.cos(δij) + Bij * np.sin(δij))
                    
                    # J21[i,j] = ∂Qi/∂δj
                    J21[i, j] = -Vi * Vj * (Gij * np.cos(δij) + Bij * np.sin(δij))
                    
                    # J22[i,j] = ∂Qi/∂Vj  
                    J22[i, j] = Vi * (Gij * np.sin(δij) - Bij * np.cos(δij))
        
        # Assemble full Jacobian matrix
        jacobian = np.block([[J11, J12], [J21, J22]])
        
        return jacobian
    
    def analyze_jacobian_properties(self, jacobian: np.ndarray) -> Dict[str, Any]:
        """
        Analyze properties of the Jacobian matrix
        
        Args:
            jacobian: Jacobian matrix
            
        Returns:
            Dictionary containing Jacobian properties
        """
        # Eigenvalue analysis
        eigenvalues = eigvals(jacobian)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        # Stability indicators
        min_real_eigenvalue = np.min(real_parts)
        max_real_eigenvalue = np.max(real_parts)
        num_negative_eigenvalues = np.sum(real_parts < 0)
        
        # Matrix conditioning
        condition_number = cond(jacobian)
        determinant = det(jacobian)
        
        # Spectral radius
        spectral_radius = np.max(np.abs(eigenvalues))
        
        # Singular value decomposition
        try:
            U, singular_values, Vt = np.linalg.svd(jacobian)
            min_singular_value = np.min(singular_values)
            max_singular_value = np.max(singular_values)
        except:
            min_singular_value = 0
            max_singular_value = 0
        
        return {
            'eigenvalues': eigenvalues,
            'real_eigenvalues': real_parts,
            'imaginary_eigenvalues': imag_parts,
            'min_real_eigenvalue': min_real_eigenvalue,
            'max_real_eigenvalue': max_real_eigenvalue,
            'num_negative_eigenvalues': num_negative_eigenvalues,
            'condition_number': condition_number,
            'determinant': determinant,
            'spectral_radius': spectral_radius,
            'singular_values': singular_values if 'singular_values' in locals() else None,
            'min_singular_value': min_singular_value,
            'max_singular_value': max_singular_value,
            'is_stable': min_real_eigenvalue > -1e-6,  # Stability criterion
            'stability_margin': abs(min_real_eigenvalue)
        }
    
    def establish_baseline(self, normal_operating_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Establish baseline Jacobian properties from normal operation
        
        Args:
            normal_operating_data: List of normal operating points, each containing:
                - bus_voltages, bus_angles, y_matrix, p_injections, q_injections
                
        Returns:
            Dictionary containing baseline properties
        """
        start_time = time.time()
        
        baseline_properties = []
        jacobians = []
        
        for data_point in normal_operating_data:
            # Calculate Jacobian for this operating point
            jacobian = self.calculate_jacobian(
                data_point['bus_voltages'],
                data_point['bus_angles'], 
                data_point['y_matrix'],
                data_point['p_injections'],
                data_point['q_injections']
            )
            
            jacobians.append(jacobian)
            
            # Analyze properties
            properties = self.analyze_jacobian_properties(jacobian)
            baseline_properties.append(properties)
        
        # Calculate baseline statistics
        condition_numbers = [props['condition_number'] for props in baseline_properties]
        min_eigenvalues = [props['min_real_eigenvalue'] for props in baseline_properties]
        spectral_radii = [props['spectral_radius'] for props in baseline_properties]
        stability_margins = [props['stability_margin'] for props in baseline_properties]
        
        self.baseline_condition_number = {
            'mean': np.mean(condition_numbers),
            'std': np.std(condition_numbers),
            'min': np.min(condition_numbers),
            'max': np.max(condition_numbers)
        }
        
        self.baseline_eigenvalues = {
            'mean_min_real': np.mean(min_eigenvalues),
            'std_min_real': np.std(min_eigenvalues),
            'mean_spectral_radius': np.mean(spectral_radii),
            'std_spectral_radius': np.std(spectral_radii)
        }
        
        # Set stability threshold (3-sigma rule)
        self.stability_threshold = (
            np.mean(stability_margins) - 3 * np.std(stability_margins)
        )
        
        self.training_time = time.time() - start_time
        
        return {
            'training_time': self.training_time,
            'num_baseline_points': len(normal_operating_data),
            'baseline_condition_number': self.baseline_condition_number,
            'baseline_eigenvalues': self.baseline_eigenvalues,
            'stability_threshold': self.stability_threshold,
            'baseline_properties': baseline_properties
        }
    
    def detect_anomalies(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect anomalies in Jacobian properties
        
        Args:
            test_data: List of test operating points
            
        Returns:
            Dictionary containing anomaly detection results
        """
        if self.baseline_condition_number is None:
            raise ValueError("Baseline not established. Call establish_baseline() first.")
        
        start_time = time.time()
        
        anomaly_results = []
        jacobian_anomalies = []
        
        for i, data_point in enumerate(test_data):
            # Calculate Jacobian
            jacobian = self.calculate_jacobian(
                data_point['bus_voltages'],
                data_point['bus_angles'],
                data_point['y_matrix'], 
                data_point['p_injections'],
                data_point['q_injections']
            )
            
            # Analyze properties
            properties = self.analyze_jacobian_properties(jacobian)
            
            # Detect anomalies
            anomalies = self._detect_jacobian_anomalies(properties)
            
            anomaly_results.append({
                'sample_index': i,
                'properties': properties,
                'anomalies': anomalies,
                'is_anomalous': any(anomalies.values()),
                'anomaly_score': sum(anomalies.values())
            })
            
            jacobian_anomalies.append(anomalies)
        
        self.prediction_time = time.time() - start_time
        
        # Summary statistics
        total_anomalies = sum([result['is_anomalous'] for result in anomaly_results])
        anomaly_rate = total_anomalies / len(test_data)
        
        return {
            'prediction_time': self.prediction_time,
            'anomaly_results': anomaly_results,
            'total_anomalies': total_anomalies,
            'anomaly_rate': anomaly_rate,
            'samples_per_second': len(test_data) / self.prediction_time if self.prediction_time > 0 else np.inf
        }
    
    def _detect_jacobian_anomalies(self, properties: Dict[str, Any]) -> Dict[str, bool]:
        """
        Detect specific types of Jacobian anomalies
        
        Args:
            properties: Jacobian properties
            
        Returns:
            Dictionary of anomaly flags
        """
        anomalies = {}
        
        # Condition number anomaly
        cond_threshold = (
            self.baseline_condition_number['mean'] + 
            3 * self.baseline_condition_number['std']
        )
        anomalies['high_condition_number'] = properties['condition_number'] > cond_threshold
        
        # Eigenvalue anomaly
        eigenvalue_threshold = (
            self.baseline_eigenvalues['mean_min_real'] - 
            3 * self.baseline_eigenvalues['std_min_real']
        )
        anomalies['unstable_eigenvalues'] = properties['min_real_eigenvalue'] < eigenvalue_threshold
        
        # Spectral radius anomaly
        spectral_threshold = (
            self.baseline_eigenvalues['mean_spectral_radius'] + 
            3 * self.baseline_eigenvalues['std_spectral_radius']
        )
        anomalies['high_spectral_radius'] = properties['spectral_radius'] > spectral_threshold
        
        # Stability margin anomaly
        anomalies['low_stability_margin'] = properties['stability_margin'] < self.stability_threshold
        
        # Singularity detection
        anomalies['near_singular'] = properties['condition_number'] > 1e12
        
        # Determinant anomaly (near zero indicates singularity)
        anomalies['small_determinant'] = abs(properties['determinant']) < 1e-10
        
        return anomalies
    
    def monitor_real_time(self, data_stream, window_size: int = 50, 
                         alert_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Real-time monitoring of Jacobian properties
        
        Args:
            data_stream: Stream of operating point data
            window_size: Size of sliding window for analysis
            alert_threshold: Threshold for anomaly rate to trigger alert
            
        Returns:
            Dictionary containing real-time monitoring results
        """
        monitoring_results = {
            'timestamps': [],
            'anomaly_flags': [],
            'condition_numbers': [],
            'stability_margins': [],
            'alerts': []
        }
        
        window_buffer = []
        
        for timestamp, data_point in data_stream:
            # Calculate Jacobian and properties
            jacobian = self.calculate_jacobian(
                data_point['bus_voltages'],
                data_point['bus_angles'],
                data_point['y_matrix'],
                data_point['p_injections'], 
                data_point['q_injections']
            )
            
            properties = self.analyze_jacobian_properties(jacobian)
            anomalies = self._detect_jacobian_anomalies(properties)
            
            # Update monitoring results
            monitoring_results['timestamps'].append(timestamp)
            monitoring_results['anomaly_flags'].append(any(anomalies.values()))
            monitoring_results['condition_numbers'].append(properties['condition_number'])
            monitoring_results['stability_margins'].append(properties['stability_margin'])
            
            # Sliding window analysis
            window_buffer.append(any(anomalies.values()))
            if len(window_buffer) > window_size:
                window_buffer.pop(0)
            
            # Check for alerts
            if len(window_buffer) >= window_size:
                anomaly_rate = sum(window_buffer) / len(window_buffer)
                if anomaly_rate > alert_threshold:
                    alert = {
                        'timestamp': timestamp,
                        'anomaly_rate': anomaly_rate,
                        'alert_type': 'high_anomaly_rate',
                        'severity': 'high' if anomaly_rate > 0.2 else 'medium'
                    }
                    monitoring_results['alerts'].append(alert)
                    logger.warning(f"Alert: High anomaly rate {anomaly_rate:.3f} at {timestamp}")
        
        return monitoring_results
    
    def visualize_eigenvalues(self, jacobian: np.ndarray, save_path: Optional[str] = None):
        """
        Visualize eigenvalues in the complex plane
        
        Args:
            jacobian: Jacobian matrix
            save_path: Path to save the plot
        """
        eigenvalues = eigvals(jacobian)
        
        plt.figure(figsize=(10, 8))
        
        # Plot eigenvalues
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        plt.scatter(real_parts, imag_parts, c='blue', s=50, alpha=0.7)
        
        # Add stability boundary (left half-plane)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                   label='Stability Boundary')
        
        # Highlight unstable eigenvalues
        unstable_mask = real_parts > 0
        if np.any(unstable_mask):
            plt.scatter(real_parts[unstable_mask], imag_parts[unstable_mask], 
                       c='red', s=100, marker='x', linewidth=3,
                       label='Unstable Eigenvalues')
        
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Jacobian Eigenvalues in Complex Plane')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations for critical eigenvalues
        min_real_idx = np.argmin(real_parts)
        plt.annotate(f'λ_min = {eigenvalues[min_real_idx]:.3f}',
                    xy=(real_parts[min_real_idx], imag_parts[min_real_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_stability_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive stability analysis report
        
        Args:
            test_results: Results from detect_anomalies()
            
        Returns:
            Formatted stability report
        """
        report = []
        report.append("=" * 60)
        report.append("JACOBIAN STABILITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        total_samples = len(test_results['anomaly_results'])
        anomalous_samples = test_results['total_anomalies']
        anomaly_rate = test_results['anomaly_rate']
        
        report.append(f"Total Samples Analyzed: {total_samples}")
        report.append(f"Anomalous Samples: {anomalous_samples}")
        report.append(f"Anomaly Rate: {anomaly_rate:.3f} ({anomaly_rate*100:.1f}%)")
        report.append(f"Analysis Time: {test_results['prediction_time']:.3f} seconds")
        report.append(f"Processing Rate: {test_results['samples_per_second']:.1f} samples/second")
        report.append("")
        
        # Anomaly breakdown
        anomaly_types = ['high_condition_number', 'unstable_eigenvalues', 
                        'high_spectral_radius', 'low_stability_margin',
                        'near_singular', 'small_determinant']
        
        report.append("ANOMALY TYPE BREAKDOWN:")
        report.append("-" * 30)
        
        for anomaly_type in anomaly_types:
            count = sum([1 for result in test_results['anomaly_results'] 
                        if result['anomalies'].get(anomaly_type, False)])
            percentage = (count / total_samples) * 100
            report.append(f"{anomaly_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        report.append("")
        
        # Critical cases
        critical_cases = [result for result in test_results['anomaly_results'] 
                         if result['anomaly_score'] >= 3]
        
        if critical_cases:
            report.append("CRITICAL STABILITY ISSUES:")
            report.append("-" * 30)
            for case in critical_cases[:5]:  # Show top 5
                report.append(f"Sample {case['sample_index']}: "
                            f"Anomaly Score = {case['anomaly_score']}")
                report.append(f"  Condition Number: {case['properties']['condition_number']:.2e}")
                report.append(f"  Min Real Eigenvalue: {case['properties']['min_real_eigenvalue']:.6f}")
                report.append(f"  Stability Margin: {case['properties']['stability_margin']:.6f}")
                report.append("")
        
        return "\n".join(report)