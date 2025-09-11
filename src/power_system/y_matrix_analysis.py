"""
Y-Matrix (Admittance Matrix) Consistency Checking for Topology Verification
Implements Y-matrix analysis for detecting topology manipulation attacks
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.linalg import eigvals, norm
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional
import logging
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YMatrixAnalyzer:
    """
    Y-matrix analysis for power system topology verification and attack detection
    """
    
    def __init__(self):
        self.baseline_y_matrix = None
        self.baseline_properties = None
        self.topology_graph = None
        self.training_time = 0
        self.prediction_time = 0
        
    def build_y_matrix(self, bus_data: pd.DataFrame, line_data: pd.DataFrame) -> np.ndarray:
        """
        Build admittance matrix from bus and line data
        
        Args:
            bus_data: DataFrame with bus information (bus_id, voltage_base, etc.)
            line_data: DataFrame with line information (from_bus, to_bus, resistance, reactance, susceptance)
            
        Returns:
            Y-matrix (admittance matrix)
        """
        n_buses = len(bus_data)
        y_matrix = np.zeros((n_buses, n_buses), dtype=complex)
        
        # Create bus mapping
        bus_mapping = {bus_id: idx for idx, bus_id in enumerate(bus_data['bus_id'])}
        
        # Add line admittances
        for _, line in line_data.iterrows():
            from_bus = bus_mapping[line['from_bus']]
            to_bus = bus_mapping[line['to_bus']]
            
            # Line parameters
            r = line['resistance']
            x = line['reactance']
            b = line.get('susceptance', 0.0)  # Line charging susceptance
            
            # Line admittance
            z = complex(r, x)
            y_line = 1 / z if abs(z) > 1e-10 else 0
            
            # Off-diagonal elements (negative of line admittance)
            y_matrix[from_bus, to_bus] -= y_line
            y_matrix[to_bus, from_bus] -= y_line
            
            # Diagonal elements (sum of connected line admittances + shunt)
            y_matrix[from_bus, from_bus] += y_line + 1j * b / 2
            y_matrix[to_bus, to_bus] += y_line + 1j * b / 2
        
        # Add shunt admittances (if any)
        for _, bus in bus_data.iterrows():
            bus_idx = bus_mapping[bus['bus_id']]
            if 'shunt_conductance' in bus:
                y_matrix[bus_idx, bus_idx] += complex(
                    bus.get('shunt_conductance', 0),
                    bus.get('shunt_susceptance', 0)
                )
        
        return y_matrix
    
    def analyze_y_matrix_properties(self, y_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze properties of the Y-matrix
        
        Args:
            y_matrix: Admittance matrix
            
        Returns:
            Dictionary containing Y-matrix properties
        """
        # Basic properties
        n_buses = y_matrix.shape[0]
        
        # Eigenvalue analysis
        eigenvalues = eigvals(y_matrix)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        # Matrix norms
        frobenius_norm = norm(y_matrix, 'fro')
        spectral_norm = norm(y_matrix, 2)
        
        # Sparsity analysis
        non_zero_elements = np.count_nonzero(y_matrix)
        sparsity = 1 - (non_zero_elements / (n_buses ** 2))
        
        # Diagonal dominance check
        diagonal_elements = np.diag(y_matrix)
        off_diagonal_sums = np.sum(np.abs(y_matrix), axis=1) - np.abs(diagonal_elements)
        diagonal_dominance = np.all(np.abs(diagonal_elements) >= off_diagonal_sums)
        
        # Symmetry check (Y-matrix should be symmetric)
        symmetry_error = norm(y_matrix - y_matrix.T, 'fro')
        is_symmetric = symmetry_error < 1e-10
        
        # Connectivity analysis
        adjacency_matrix = (np.abs(y_matrix) > 1e-10).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-connections
        
        # Create graph for connectivity analysis
        G = nx.from_numpy_array(adjacency_matrix)
        is_connected = nx.is_connected(G)
        num_components = nx.number_connected_components(G)
        
        # Degree analysis
        degrees = dict(G.degree())
        avg_degree = np.mean(list(degrees.values()))
        max_degree = max(degrees.values()) if degrees else 0
        min_degree = min(degrees.values()) if degrees else 0
        
        # Rank analysis
        rank = np.linalg.matrix_rank(y_matrix)
        is_full_rank = rank == n_buses
        
        return {
            'n_buses': n_buses,
            'eigenvalues': eigenvalues,
            'real_eigenvalues': real_parts,
            'imaginary_eigenvalues': imag_parts,
            'min_real_eigenvalue': np.min(real_parts),
            'max_real_eigenvalue': np.max(real_parts),
            'frobenius_norm': frobenius_norm,
            'spectral_norm': spectral_norm,
            'sparsity': sparsity,
            'non_zero_elements': non_zero_elements,
            'diagonal_dominance': diagonal_dominance,
            'is_symmetric': is_symmetric,
            'symmetry_error': symmetry_error,
            'is_connected': is_connected,
            'num_components': num_components,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'min_degree': min_degree,
            'rank': rank,
            'is_full_rank': is_full_rank,
            'condition_number': np.linalg.cond(y_matrix)
        }
    
    def establish_baseline(self, bus_data: pd.DataFrame, line_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Establish baseline Y-matrix from normal topology
        
        Args:
            bus_data: Normal bus configuration
            line_data: Normal line configuration
            
        Returns:
            Dictionary containing baseline properties
        """
        start_time = time.time()
        
        # Build baseline Y-matrix
        self.baseline_y_matrix = self.build_y_matrix(bus_data, line_data)
        
        # Analyze baseline properties
        self.baseline_properties = self.analyze_y_matrix_properties(self.baseline_y_matrix)
        
        # Create topology graph
        adjacency_matrix = (np.abs(self.baseline_y_matrix) > 1e-10).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        self.topology_graph = nx.from_numpy_array(adjacency_matrix)
        
        # Add node labels
        node_mapping = {i: f"Bus_{bus_data.iloc[i]['bus_id']}" for i in range(len(bus_data))}
        self.topology_graph = nx.relabel_nodes(self.topology_graph, node_mapping)
        
        self.training_time = time.time() - start_time
        
        return {
            'training_time': self.training_time,
            'baseline_y_matrix': self.baseline_y_matrix,
            'baseline_properties': self.baseline_properties,
            'topology_graph': self.topology_graph
        }
    
    def detect_topology_attacks(self, test_bus_data: pd.DataFrame, 
                               test_line_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect topology manipulation attacks by comparing Y-matrices
        
        Args:
            test_bus_data: Test bus configuration
            test_line_data: Test line configuration
            
        Returns:
            Dictionary containing attack detection results
        """
        if self.baseline_y_matrix is None:
            raise ValueError("Baseline not established. Call establish_baseline() first.")
        
        start_time = time.time()
        
        # Build test Y-matrix
        test_y_matrix = self.build_y_matrix(test_bus_data, test_line_data)
        
        # Analyze test properties
        test_properties = self.analyze_y_matrix_properties(test_y_matrix)
        
        # Compare with baseline
        attack_indicators = self._detect_y_matrix_anomalies(test_y_matrix, test_properties)
        
        # Detailed topology comparison
        topology_changes = self._analyze_topology_changes(test_y_matrix)
        
        self.prediction_time = time.time() - start_time
        
        return {
            'prediction_time': self.prediction_time,
            'test_y_matrix': test_y_matrix,
            'test_properties': test_properties,
            'attack_indicators': attack_indicators,
            'topology_changes': topology_changes,
            'is_attack_detected': any(attack_indicators.values()),
            'attack_severity': sum([1 for v in attack_indicators.values() if v])
        }
    
    def _detect_y_matrix_anomalies(self, test_y_matrix: np.ndarray, 
                                  test_properties: Dict[str, Any]) -> Dict[str, bool]:
        """
        Detect anomalies in Y-matrix properties
        
        Args:
            test_y_matrix: Test Y-matrix
            test_properties: Test Y-matrix properties
            
        Returns:
            Dictionary of attack indicators
        """
        indicators = {}
        baseline = self.baseline_properties
        
        # Matrix difference analysis
        y_diff = test_y_matrix - self.baseline_y_matrix
        diff_norm = norm(y_diff, 'fro')
        relative_diff = diff_norm / norm(self.baseline_y_matrix, 'fro')
        
        # Threshold-based detection (can be tuned)
        indicators['significant_matrix_change'] = relative_diff > 0.01  # 1% change threshold
        
        # Connectivity changes
        indicators['connectivity_change'] = (
            test_properties['is_connected'] != baseline['is_connected'] or
            test_properties['num_components'] != baseline['num_components']
        )
        
        # Sparsity changes
        sparsity_change = abs(test_properties['sparsity'] - baseline['sparsity'])
        indicators['sparsity_change'] = sparsity_change > 0.05  # 5% change threshold
        
        # Degree distribution changes
        degree_change = abs(test_properties['avg_degree'] - baseline['avg_degree'])
        indicators['degree_distribution_change'] = degree_change > 1.0
        
        # Symmetry violation
        indicators['symmetry_violation'] = not test_properties['is_symmetric']
        
        # Rank changes
        indicators['rank_change'] = test_properties['rank'] != baseline['rank']
        
        # Eigenvalue changes
        eigenvalue_diff = abs(test_properties['min_real_eigenvalue'] - baseline['min_real_eigenvalue'])
        indicators['eigenvalue_change'] = eigenvalue_diff > 0.1
        
        # Condition number changes
        cond_ratio = test_properties['condition_number'] / baseline['condition_number']
        indicators['condition_number_change'] = cond_ratio > 2.0 or cond_ratio < 0.5
        
        return indicators
    
    def _analyze_topology_changes(self, test_y_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze specific topology changes
        
        Args:
            test_y_matrix: Test Y-matrix
            
        Returns:
            Dictionary containing topology change analysis
        """
        # Create test topology graph
        test_adjacency = (np.abs(test_y_matrix) > 1e-10).astype(int)
        np.fill_diagonal(test_adjacency, 0)
        test_graph = nx.from_numpy_array(test_adjacency)
        
        baseline_adjacency = (np.abs(self.baseline_y_matrix) > 1e-10).astype(int)
        np.fill_diagonal(baseline_adjacency, 0)
        
        # Find added and removed connections
        added_connections = []
        removed_connections = []
        
        n_buses = test_y_matrix.shape[0]
        for i in range(n_buses):
            for j in range(i+1, n_buses):
                baseline_connected = baseline_adjacency[i, j] == 1
                test_connected = test_adjacency[i, j] == 1
                
                if test_connected and not baseline_connected:
                    added_connections.append((i, j))
                elif baseline_connected and not test_connected:
                    removed_connections.append((i, j))
        
        # Analyze impedance changes
        impedance_changes = []
        for i in range(n_buses):
            for j in range(n_buses):
                if abs(self.baseline_y_matrix[i, j]) > 1e-10 and abs(test_y_matrix[i, j]) > 1e-10:
                    baseline_val = self.baseline_y_matrix[i, j]
                    test_val = test_y_matrix[i, j]
                    relative_change = abs(test_val - baseline_val) / abs(baseline_val)
                    
                    if relative_change > 0.1:  # 10% change threshold
                        impedance_changes.append({
                            'bus_pair': (i, j),
                            'baseline_value': baseline_val,
                            'test_value': test_val,
                            'relative_change': relative_change
                        })
        
        return {
            'added_connections': added_connections,
            'removed_connections': removed_connections,
            'num_added_connections': len(added_connections),
            'num_removed_connections': len(removed_connections),
            'impedance_changes': impedance_changes,
            'num_impedance_changes': len(impedance_changes)
        }
    
    def visualize_topology_comparison(self, test_y_matrix: np.ndarray, 
                                    save_path: Optional[str] = None):
        """
        Visualize baseline and test topologies for comparison
        
        Args:
            test_y_matrix: Test Y-matrix
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Baseline topology
        pos = nx.spring_layout(self.topology_graph, seed=42)
        nx.draw(self.topology_graph, pos, ax=ax1, with_labels=True, 
                node_color='lightblue', node_size=500, font_size=8,
                edge_color='gray', width=1.5)
        ax1.set_title('Baseline Topology')
        
        # Test topology
        test_adjacency = (np.abs(test_y_matrix) > 1e-10).astype(int)
        np.fill_diagonal(test_adjacency, 0)
        test_graph = nx.from_numpy_array(test_adjacency)
        
        # Use same positions for comparison
        nx.draw(test_graph, pos, ax=ax2, with_labels=True,
                node_color='lightcoral', node_size=500, font_size=8,
                edge_color='gray', width=1.5)
        ax2.set_title('Test Topology')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_y_matrix_heatmap(self, y_matrix: np.ndarray, title: str = "Y-Matrix",
                                  save_path: Optional[str] = None):
        """
        Visualize Y-matrix as a heatmap
        
        Args:
            y_matrix: Y-matrix to visualize
            title: Plot title
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Real part
        im1 = ax1.imshow(np.real(y_matrix), cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'{title} - Real Part')
        ax1.set_xlabel('Bus Index')
        ax1.set_ylabel('Bus Index')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Imaginary part
        im2 = ax2.imshow(np.imag(y_matrix), cmap='RdBu_r', aspect='auto')
        ax2.set_title(f'{title} - Imaginary Part')
        ax2.set_xlabel('Bus Index')
        ax2.set_ylabel('Bus Index')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_topology_report(self, detection_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive topology analysis report
        
        Args:
            detection_results: Results from detect_topology_attacks()
            
        Returns:
            Formatted topology analysis report
        """
        report = []
        report.append("=" * 60)
        report.append("TOPOLOGY ATTACK DETECTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        is_attack = detection_results['is_attack_detected']
        severity = detection_results['attack_severity']
        
        report.append(f"Attack Detected: {'YES' if is_attack else 'NO'}")
        report.append(f"Attack Severity: {severity}/8")
        report.append(f"Analysis Time: {detection_results['prediction_time']:.3f} seconds")
        report.append("")
        
        # Attack indicators
        report.append("ATTACK INDICATORS:")
        report.append("-" * 20)
        
        indicators = detection_results['attack_indicators']
        for indicator, detected in indicators.items():
            status = "DETECTED" if detected else "NORMAL"
            report.append(f"{indicator.replace('_', ' ').title()}: {status}")
        
        report.append("")
        
        # Topology changes
        changes = detection_results['topology_changes']
        
        report.append("TOPOLOGY CHANGES:")
        report.append("-" * 20)
        report.append(f"Added Connections: {changes['num_added_connections']}")
        report.append(f"Removed Connections: {changes['num_removed_connections']}")
        report.append(f"Impedance Changes: {changes['num_impedance_changes']}")
        
        if changes['added_connections']:
            report.append("\nAdded Connections:")
            for connection in changes['added_connections'][:5]:  # Show first 5
                report.append(f"  Bus {connection[0]} - Bus {connection[1]}")
        
        if changes['removed_connections']:
            report.append("\nRemoved Connections:")
            for connection in changes['removed_connections'][:5]:  # Show first 5
                report.append(f"  Bus {connection[0]} - Bus {connection[1]}")
        
        if changes['impedance_changes']:
            report.append("\nSignificant Impedance Changes:")
            for change in changes['impedance_changes'][:5]:  # Show first 5
                report.append(f"  Bus {change['bus_pair'][0]}-{change['bus_pair'][1]}: "
                            f"{change['relative_change']:.1%} change")
        
        return "\n".join(report)