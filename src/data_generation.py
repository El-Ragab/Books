"""
Data Generation Framework for Cyber-Attack Detection Research
Generates synthetic power system data with normal operation and attack scenarios
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from power_system.ieee_test_systems import get_ieee_test_system
from power_system.jacobian_analysis import JacobianAnalyzer
from power_system.y_matrix_analysis import YMatrixAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PowerSystemDataGenerator:
    """
    Comprehensive data generator for power system cyber-attack detection research
    """
    
    def __init__(self, system_name: str = '14bus', sampling_rate: float = 1000.0):
        self.system_name = system_name
        self.sampling_rate = sampling_rate  # Hz
        self.test_system = get_ieee_test_system(system_name)
        self.jacobian_analyzer = JacobianAnalyzer()
        self.y_matrix_analyzer = YMatrixAnalyzer()
        
        # Get base system data
        self.base_system_data = self.test_system.get_system_data()
        self.attack_scenarios = self.test_system.create_attack_scenarios()
        
        # Initialize analyzers with baseline data
        self._initialize_analyzers()
        
    def _initialize_analyzers(self):
        """Initialize power system analyzers with baseline data"""
        bus_data = self.base_system_data['bus_data']
        line_data = self.base_system_data['line_data']
        
        # Initialize Y-matrix analyzer
        self.y_matrix_analyzer.establish_baseline(bus_data, line_data)
        
        logger.info("Power system analyzers initialized")
    
    def generate_normal_operation_data(self, duration_hours: float = 24.0, 
                                     noise_level: float = 0.05) -> Dict[str, Any]:
        """
        Generate normal operation data with realistic variations
        
        Args:
            duration_hours: Duration of data generation in hours
            noise_level: Level of measurement noise (0-1)
            
        Returns:
            Dictionary containing normal operation data
        """
        logger.info(f"Generating {duration_hours} hours of normal operation data")
        
        # Time parameters
        dt = 1.0 / self.sampling_rate
        n_samples = int(duration_hours * 3600 * self.sampling_rate)
        time_vector = np.arange(n_samples) * dt
        
        # Base system parameters
        bus_data = self.base_system_data['bus_data'].copy()
        n_buses = len(bus_data)
        
        # Initialize data arrays
        data = {
            'timestamps': time_vector,
            'bus_voltages': np.zeros((n_samples, n_buses)),
            'bus_angles': np.zeros((n_samples, n_buses)),
            'bus_frequencies': np.zeros((n_samples, n_buses)),
            'p_injections': np.zeros((n_samples, n_buses)),
            'q_injections': np.zeros((n_samples, n_buses)),
            'line_currents': [],
            'protection_signals': np.zeros((n_samples, n_buses * 3)),  # 3 protection zones per bus
            'pmu_data': np.zeros((n_samples, n_buses * 6)),  # V_mag, V_angle, I_mag, I_angle, freq, ROCOF
            'ied_communications': []
        }
        
        # Generate base load profile (daily pattern)
        daily_pattern = self._generate_daily_load_pattern(time_vector / 3600)  # Convert to hours
        
        for i, t in enumerate(time_vector):
            # Time-varying load pattern
            load_multiplier = daily_pattern[i] * (1 + noise_level * np.random.normal(0, 1, n_buses))
            
            # Bus voltages with small variations
            base_voltages = bus_data['voltage_magnitude'].values
            voltage_variations = noise_level * np.random.normal(0, 0.02, n_buses)
            data['bus_voltages'][i] = base_voltages + voltage_variations
            
            # Bus angles with small variations
            base_angles = bus_data['voltage_angle'].values * np.pi / 180  # Convert to radians
            angle_variations = noise_level * np.random.normal(0, 0.01, n_buses)
            data['bus_angles'][i] = base_angles + angle_variations
            
            # Frequency variations around nominal (60 Hz)
            freq_variations = noise_level * np.random.normal(0, 0.1, n_buses)
            data['bus_frequencies'][i] = 60.0 + freq_variations
            
            # Power injections with load variations
            base_p = bus_data['p_load'].values * load_multiplier
            p_variations = noise_level * np.random.normal(0, 0.05 * np.abs(base_p) + 1e-3)
            data['p_injections'][i] = -(base_p + p_variations)  # Negative for loads
            
            base_q = bus_data['q_load'].values * load_multiplier
            q_variations = noise_level * np.random.normal(0, 0.05 * np.abs(base_q) + 1e-3)
            data['q_injections'][i] = -(base_q + q_variations)
            
            # Protection signals (distance relay measurements)
            protection_signals = self._generate_protection_signals(
                data['bus_voltages'][i], 
                data['bus_angles'][i],
                noise_level
            )
            data['protection_signals'][i] = protection_signals
            
            # PMU data
            pmu_signals = self._generate_pmu_data(
                data['bus_voltages'][i],
                data['bus_angles'][i], 
                data['bus_frequencies'][i],
                noise_level
            )
            data['pmu_data'][i] = pmu_signals
            
            # Generate IED communication data periodically
            if i % int(self.sampling_rate / 10) == 0:  # Every 100ms
                ied_comm = self._generate_ied_communication(i, data)
                data['ied_communications'].append(ied_comm)
        
        # Add metadata
        data['metadata'] = {
            'system_name': self.system_name,
            'duration_hours': duration_hours,
            'sampling_rate': self.sampling_rate,
            'noise_level': noise_level,
            'n_buses': n_buses,
            'data_type': 'normal_operation',
            'generation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Generated {n_samples} samples of normal operation data")
        return data
    
    def generate_attack_data(self, attack_scenario: str, duration_hours: float = 1.0,
                           attack_start_time: float = 0.3, noise_level: float = 0.05) -> Dict[str, Any]:
        """
        Generate data with cyber-attack scenarios
        
        Args:
            attack_scenario: Name of attack scenario
            duration_hours: Total duration in hours
            attack_start_time: When attack starts (fraction of total duration)
            noise_level: Level of measurement noise
            
        Returns:
            Dictionary containing attack scenario data
        """
        if attack_scenario not in self.attack_scenarios:
            raise ValueError(f"Unknown attack scenario: {attack_scenario}")
        
        logger.info(f"Generating attack scenario: {attack_scenario}")
        
        # Generate normal data first
        data = self.generate_normal_operation_data(duration_hours, noise_level)
        
        # Attack parameters
        attack_start_sample = int(attack_start_time * len(data['timestamps']))
        attack_data_config = self.attack_scenarios[attack_scenario]
        attack_type = attack_data_config['attack_type']
        
        # Apply attack based on type
        if attack_type == 'line_outage':
            data = self._apply_line_outage_attack(data, attack_start_sample, attack_data_config)
        elif attack_type == 'load_manipulation':
            data = self._apply_load_manipulation_attack(data, attack_start_sample, attack_data_config)
        elif attack_type == 'generator_setpoint':
            data = self._apply_generator_attack(data, attack_start_sample, attack_data_config)
        elif attack_type == 'impedance_manipulation':
            data = self._apply_impedance_attack(data, attack_start_sample, attack_data_config)
        elif attack_type in ['coordinated_load_attack', 'large_scale_coordinated_attack']:
            data = self._apply_coordinated_attack(data, attack_start_sample, attack_data_config)
        elif attack_type == 'false_data_injection':
            data = self._apply_false_data_injection(data, attack_start_sample, attack_data_config)
        elif attack_type == 'denial_of_service':
            data = self._apply_dos_attack(data, attack_start_sample, attack_data_config)
        
        # Update metadata
        data['metadata']['data_type'] = 'attack_scenario'
        data['metadata']['attack_scenario'] = attack_scenario
        data['metadata']['attack_type'] = attack_type
        data['metadata']['attack_start_time'] = attack_start_time
        data['metadata']['attack_start_sample'] = attack_start_sample
        
        # Add ground truth labels
        labels = np.zeros(len(data['timestamps']))
        labels[attack_start_sample:] = 1  # 1 for attack, 0 for normal
        data['labels'] = labels
        
        logger.info(f"Generated attack scenario data with {np.sum(labels)} attack samples")
        return data
    
    def _generate_daily_load_pattern(self, hours: np.ndarray) -> np.ndarray:
        """Generate realistic daily load pattern"""
        # Typical daily load curve
        pattern = (0.7 + 0.3 * np.sin(2 * np.pi * (hours - 6) / 24) +
                  0.1 * np.sin(4 * np.pi * (hours - 6) / 24))
        return np.clip(pattern, 0.5, 1.3)  # Reasonable bounds
    
    def _generate_protection_signals(self, voltages: np.ndarray, angles: np.ndarray,
                                   noise_level: float) -> np.ndarray:
        """Generate protection relay signals (impedance measurements)"""
        n_buses = len(voltages)
        signals = np.zeros(n_buses * 3)  # 3 zones per bus
        
        for i in range(n_buses):
            # Zone 1, 2, 3 impedance measurements
            base_impedance = np.random.uniform(0.5, 2.0, 3)
            noise = noise_level * np.random.normal(0, 0.1, 3)
            
            # Impedance depends on voltage and current
            voltage_factor = voltages[i] / 1.0  # Normalized
            impedance = base_impedance * voltage_factor + noise
            
            signals[i*3:(i+1)*3] = impedance
        
        return signals
    
    def _generate_pmu_data(self, voltages: np.ndarray, angles: np.ndarray,
                          frequencies: np.ndarray, noise_level: float) -> np.ndarray:
        """Generate PMU measurement data"""
        n_buses = len(voltages)
        pmu_data = np.zeros(n_buses * 6)
        
        for i in range(n_buses):
            # PMU measurements: V_mag, V_angle, I_mag, I_angle, freq, ROCOF
            v_mag = voltages[i] + noise_level * np.random.normal(0, 0.001)
            v_angle = angles[i] + noise_level * np.random.normal(0, 0.001)
            
            # Current magnitude and angle (simplified)
            i_mag = np.random.uniform(0.5, 2.0) + noise_level * np.random.normal(0, 0.01)
            i_angle = angles[i] - np.pi/6 + noise_level * np.random.normal(0, 0.01)
            
            freq = frequencies[i] + noise_level * np.random.normal(0, 0.001)
            rocof = noise_level * np.random.normal(0, 0.1)  # Rate of change of frequency
            
            pmu_data[i*6:(i+1)*6] = [v_mag, v_angle, i_mag, i_angle, freq, rocof]
        
        return pmu_data
    
    def _generate_ied_communication(self, sample_idx: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate IED communication data"""
        timestamp = data['timestamps'][sample_idx]
        
        # Sample IED communication message
        comm_data = {
            'timestamp': timestamp,
            'message_type': 'GOOSE',  # IEC 61850 GOOSE message
            'source_ied': f'IED_{np.random.randint(1, 10)}',
            'destination_ied': 'BROADCAST',
            'data_values': {
                'protection_status': np.random.choice(['OK', 'ALARM', 'TRIP']),
                'measurement_quality': np.random.choice(['GOOD', 'INVALID', 'QUESTIONABLE']),
                'timestamp': timestamp,
                'sequence_number': sample_idx // int(self.sampling_rate / 10)
            }
        }
        
        return comm_data
    
    def _apply_line_outage_attack(self, data: Dict[str, Any], start_sample: int, 
                                 attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply line outage attack effects"""
        # Simulate voltage and angle changes due to line outage
        for i in range(start_sample, len(data['timestamps'])):
            # Voltage drops at affected buses
            affected_buses = [0, 1]  # Buses connected to outaged line
            for bus_idx in affected_buses:
                data['bus_voltages'][i, bus_idx] *= 0.95  # 5% voltage drop
                data['bus_angles'][i, bus_idx] += 0.05  # Angle shift
            
            # Change in power flows
            data['p_injections'][i] *= 1.1  # Increased loading on remaining lines
        
        return data
    
    def _apply_load_manipulation_attack(self, data: Dict[str, Any], start_sample: int,
                                       attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply load manipulation attack"""
        # Extract attack parameters from config
        bus_data = attack_config['bus_data']
        original_bus_data = self.base_system_data['bus_data']
        
        for i in range(start_sample, len(data['timestamps'])):
            # Find buses with load changes
            for idx, (_, bus) in enumerate(bus_data.iterrows()):
                original_p = original_bus_data.iloc[idx]['p_load']
                attack_p = bus['p_load']
                
                if abs(attack_p - original_p) > 1e-6:  # Load changed
                    load_increase = attack_p / original_p if original_p > 0 else 1.5
                    
                    # Apply effects
                    data['p_injections'][i, idx] *= load_increase
                    data['q_injections'][i, idx] *= load_increase
                    data['bus_voltages'][i, idx] *= 0.98  # Voltage drop due to increased load
        
        return data
    
    def _apply_generator_attack(self, data: Dict[str, Any], start_sample: int,
                               attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply generator setpoint attack"""
        for i in range(start_sample, len(data['timestamps'])):
            # Voltage setpoint manipulation affects voltage profile
            data['bus_voltages'][i, 1] = 0.95  # Force voltage to attack setpoint
            
            # Cascading effects on neighboring buses
            for bus_idx in range(len(data['bus_voltages'][i])):
                if bus_idx != 1:
                    data['bus_voltages'][i, bus_idx] *= 0.98
        
        return data
    
    def _apply_impedance_attack(self, data: Dict[str, Any], start_sample: int,
                               attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply line impedance manipulation attack"""
        for i in range(start_sample, len(data['timestamps'])):
            # Impedance changes affect power flows and voltages
            # Simulate effects of impedance manipulation
            data['bus_voltages'][i] *= np.random.uniform(0.98, 1.02, len(data['bus_voltages'][i]))
            data['p_injections'][i] *= np.random.uniform(0.95, 1.05, len(data['p_injections'][i]))
        
        return data
    
    def _apply_coordinated_attack(self, data: Dict[str, Any], start_sample: int,
                                 attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply coordinated multi-bus attack"""
        for i in range(start_sample, len(data['timestamps'])):
            # Multiple buses affected simultaneously
            attack_buses = list(range(2, 8))  # Example coordinated attack
            
            for bus_idx in attack_buses:
                if bus_idx < len(data['bus_voltages'][i]):
                    data['p_injections'][i, bus_idx] *= 1.3  # 30% load increase
                    data['q_injections'][i, bus_idx] *= 1.3
                    data['bus_voltages'][i, bus_idx] *= 0.97  # Voltage drop
        
        return data
    
    def _apply_false_data_injection(self, data: Dict[str, Any], start_sample: int,
                                   attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply false data injection attack"""
        for i in range(start_sample, len(data['timestamps'])):
            # Inject false measurements
            false_data_buses = [2, 5, 8]
            
            for bus_idx in false_data_buses:
                if bus_idx < len(data['pmu_data'][i]) // 6:
                    # Corrupt PMU measurements
                    pmu_start = bus_idx * 6
                    data['pmu_data'][i, pmu_start] += 0.1  # False voltage magnitude
                    data['pmu_data'][i, pmu_start + 1] += 0.05  # False voltage angle
        
        return data
    
    def _apply_dos_attack(self, data: Dict[str, Any], start_sample: int,
                         attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply denial of service attack"""
        # Simulate communication failures
        for i in range(len(data['ied_communications'])):
            comm = data['ied_communications'][i]
            if comm['timestamp'] >= data['timestamps'][start_sample]:
                # Simulate packet loss and delays
                if np.random.random() < 0.3:  # 30% packet loss
                    comm['data_values']['measurement_quality'] = 'INVALID'
                    comm['data_values']['protection_status'] = 'COMM_FAIL'
        
        return data
    
    def create_ml_dataset(self, normal_hours: float = 48.0, attack_scenarios: List[str] = None,
                         attack_hours: float = 2.0, window_size: int = 100) -> Dict[str, Any]:
        """
        Create machine learning dataset with features and labels
        
        Args:
            normal_hours: Hours of normal operation data
            attack_scenarios: List of attack scenarios to include
            attack_hours: Hours per attack scenario
            window_size: Window size for feature extraction
            
        Returns:
            ML-ready dataset
        """
        logger.info("Creating machine learning dataset")
        
        if attack_scenarios is None:
            attack_scenarios = list(self.attack_scenarios.keys())
        
        # Generate normal operation data
        normal_data = self.generate_normal_operation_data(normal_hours)
        
        # Generate attack data
        attack_data_list = []
        for scenario in attack_scenarios:
            attack_data = self.generate_attack_data(scenario, attack_hours)
            attack_data_list.append(attack_data)
        
        # Extract features
        features_normal = self._extract_features(normal_data, window_size)
        labels_normal = np.zeros(len(features_normal))
        
        features_attack = []
        labels_attack = []
        
        for attack_data in attack_data_list:
            features = self._extract_features(attack_data, window_size)
            labels = np.ones(len(features))  # Attack label = 1
            features_attack.append(features)
            labels_attack.append(labels)
        
        # Combine datasets
        all_features = [features_normal] + features_attack
        all_labels = [labels_normal] + labels_attack
        
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        # Create feature names
        feature_names = self._generate_feature_names()
        
        # Shuffle dataset
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        dataset = {
            'features': X,
            'labels': y,
            'feature_names': feature_names,
            'metadata': {
                'normal_hours': normal_hours,
                'attack_scenarios': attack_scenarios,
                'attack_hours': attack_hours,
                'window_size': window_size,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_normal_samples': len(labels_normal),
                'n_attack_samples': sum(len(labels) for labels in labels_attack),
                'attack_ratio': np.mean(y)
            }
        }
        
        logger.info(f"Created ML dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return dataset
    
    def _extract_features(self, data: Dict[str, Any], window_size: int) -> np.ndarray:
        """Extract features from time series data"""
        # Combine all measurement types
        measurements = np.hstack([
            data['bus_voltages'],
            data['bus_angles'], 
            data['bus_frequencies'],
            data['p_injections'],
            data['q_injections'],
            data['protection_signals'],
            data['pmu_data']
        ])
        
        features = []
        
        # Sliding window feature extraction
        for i in range(window_size, len(measurements)):
            window_data = measurements[i-window_size:i]
            
            # Statistical features
            window_features = []
            for col in range(window_data.shape[1]):
                col_data = window_data[:, col]
                
                # Time domain features
                window_features.extend([
                    np.mean(col_data),
                    np.std(col_data),
                    np.min(col_data),
                    np.max(col_data),
                    np.percentile(col_data, 25),
                    np.percentile(col_data, 75),
                    np.mean(np.diff(col_data)),  # Rate of change
                    np.std(np.diff(col_data))    # Variability of change
                ])
            
            features.append(window_features)
        
        return np.array(features)
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for the dataset"""
        n_buses = len(self.base_system_data['bus_data'])
        
        measurement_types = [
            ('voltage_mag', n_buses),
            ('voltage_angle', n_buses),
            ('frequency', n_buses),
            ('p_injection', n_buses),
            ('q_injection', n_buses),
            ('protection', n_buses * 3),
            ('pmu', n_buses * 6)
        ]
        
        feature_stats = ['mean', 'std', 'min', 'max', 'q25', 'q75', 'rate_mean', 'rate_std']
        
        feature_names = []
        for meas_type, n_meas in measurement_types:
            for i in range(n_meas):
                for stat in feature_stats:
                    feature_names.append(f"{meas_type}_{i}_{stat}")
        
        return feature_names
    
    def save_dataset(self, dataset: Dict[str, Any], filepath: str):
        """Save dataset to file"""
        np.savez_compressed(filepath, **dataset)
        logger.info(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load dataset from file"""
        data = np.load(filepath, allow_pickle=True)
        dataset = {key: data[key] for key in data.files}
        logger.info(f"Dataset loaded from {filepath}")
        return dataset