"""
IEEE Test Systems Implementation
Provides IEEE 14-bus, 39-bus, and 118-bus test systems for validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IEEE14Bus:
    """
    IEEE 14-bus test system implementation
    """
    
    def __init__(self):
        self.name = "IEEE 14-Bus Test System"
        self.base_mva = 100.0
        self.bus_data = None
        self.line_data = None
        self.generator_data = None
        self.load_data = None
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the IEEE 14-bus system data"""
        
        # Bus data: [bus_id, type, voltage_magnitude, voltage_angle, p_load, q_load, p_gen, q_gen, base_kv]
        # Type: 1=PQ, 2=PV, 3=Slack
        bus_data_array = np.array([
            [1,  3, 1.060,  0.000,   0.00,   0.00, 232.4, -16.9, 138],
            [2,  2, 1.045, -4.982,  21.70,  12.70,  40.0,  42.4, 138],
            [3,  2, 1.010, -12.725,  94.20,  19.00,   0.0,  23.4, 138],
            [4,  1, 1.019, -10.313,  47.80,  -3.90,   0.0,   0.0, 138],
            [5,  1, 1.020,  -8.774,   7.60,   1.60,   0.0,   0.0, 138],
            [6,  2, 1.070, -14.221,  11.20,   7.50,   0.0,  12.2, 138],
            [7,  1, 1.062, -13.360,   0.00,   0.00,   0.0,   0.0, 138],
            [8,  2, 1.090, -13.360,   0.00,   0.00,   0.0,  17.4, 138],
            [9,  1, 1.056, -14.938,  29.50,  16.60,   0.0,   0.0, 138],
            [10, 1, 1.051, -15.097,   9.00,   5.80,   0.0,   0.0, 138],
            [11, 1, 1.057, -14.790,   3.50,   1.80,   0.0,   0.0, 138],
            [12, 1, 1.055, -15.076,   6.10,   1.60,   0.0,   0.0, 138],
            [13, 1, 1.050, -15.156,  13.50,   5.80,   0.0,   0.0, 138],
            [14, 1, 1.036, -16.042,  14.90,   5.00,   0.0,   0.0, 138]
        ])
        
        self.bus_data = pd.DataFrame(bus_data_array, columns=[
            'bus_id', 'type', 'voltage_magnitude', 'voltage_angle', 
            'p_load', 'q_load', 'p_gen', 'q_gen', 'base_kv'
        ])
        
        # Line data: [from_bus, to_bus, resistance, reactance, susceptance, tap_ratio, phase_shift]
        line_data_array = np.array([
            [1,  2,  0.01938, 0.05917, 0.0528, 1.0, 0.0],
            [1,  5,  0.05403, 0.22304, 0.0492, 1.0, 0.0],
            [2,  3,  0.04699, 0.19797, 0.0438, 1.0, 0.0],
            [2,  4,  0.05811, 0.17632, 0.0374, 1.0, 0.0],
            [2,  5,  0.05695, 0.17388, 0.0340, 1.0, 0.0],
            [3,  4,  0.06701, 0.17103, 0.0346, 1.0, 0.0],
            [4,  5,  0.01335, 0.04211, 0.0000, 1.0, 0.0],
            [4,  7,  0.00000, 0.20912, 0.0000, 0.978, 0.0],
            [4,  9,  0.00000, 0.55618, 0.0000, 0.969, 0.0],
            [5,  6,  0.00000, 0.25202, 0.0000, 0.932, 0.0],
            [6,  11, 0.09498, 0.19890, 0.0000, 1.0, 0.0],
            [6,  12, 0.12291, 0.25581, 0.0000, 1.0, 0.0],
            [6,  13, 0.06615, 0.13027, 0.0000, 1.0, 0.0],
            [7,  8,  0.00000, 0.17615, 0.0000, 1.0, 0.0],
            [7,  9,  0.00000, 0.11001, 0.0000, 1.0, 0.0],
            [9,  10, 0.03181, 0.08450, 0.0000, 1.0, 0.0],
            [9,  14, 0.12711, 0.27038, 0.0000, 1.0, 0.0],
            [10, 11, 0.08205, 0.19207, 0.0000, 1.0, 0.0],
            [12, 13, 0.22092, 0.19988, 0.0000, 1.0, 0.0],
            [13, 14, 0.17093, 0.34802, 0.0000, 1.0, 0.0]
        ])
        
        self.line_data = pd.DataFrame(line_data_array, columns=[
            'from_bus', 'to_bus', 'resistance', 'reactance', 
            'susceptance', 'tap_ratio', 'phase_shift'
        ])
        
        # Generator data: [bus_id, p_max, p_min, q_max, q_min, voltage_setpoint, mbase]
        generator_data_array = np.array([
            [1, 332.4, 0.0, 300.0, -300.0, 1.060, 100.0],
            [2,  80.0, 0.0,  50.0,  -40.0, 1.045, 100.0],
            [3,  50.0, 0.0,  40.0,   0.0,  1.010, 100.0],
            [6,  50.0, 0.0,  24.0,  -6.0,  1.070, 100.0],
            [8,  50.0, 0.0,  24.0,  -6.0,  1.090, 100.0]
        ])
        
        self.generator_data = pd.DataFrame(generator_data_array, columns=[
            'bus_id', 'p_max', 'p_min', 'q_max', 'q_min', 'voltage_setpoint', 'mbase'
        ])
        
        # Load data (extracted from bus data for clarity)
        load_buses = self.bus_data[self.bus_data['p_load'] > 0]
        self.load_data = load_buses[['bus_id', 'p_load', 'q_load']].copy()
    
    def get_system_data(self) -> Dict[str, Any]:
        """Get complete system data"""
        return {
            'name': self.name,
            'base_mva': self.base_mva,
            'bus_data': self.bus_data,
            'line_data': self.line_data,
            'generator_data': self.generator_data,
            'load_data': self.load_data
        }
    
    def create_attack_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Create various attack scenarios for testing"""
        base_data = self.get_system_data()
        scenarios = {}
        
        # Scenario 1: Line outage attack (remove line 1-2)
        line_data_attack1 = self.line_data[~((self.line_data['from_bus'] == 1) & 
                                           (self.line_data['to_bus'] == 2))].copy()
        scenarios['line_outage_1_2'] = {
            **base_data,
            'line_data': line_data_attack1,
            'attack_type': 'line_outage',
            'description': 'Line 1-2 outage attack'
        }
        
        # Scenario 2: Load manipulation attack (increase load at bus 9)
        bus_data_attack2 = self.bus_data.copy()
        bus_data_attack2.loc[bus_data_attack2['bus_id'] == 9, 'p_load'] *= 1.5
        bus_data_attack2.loc[bus_data_attack2['bus_id'] == 9, 'q_load'] *= 1.5
        scenarios['load_manipulation_bus9'] = {
            **base_data,
            'bus_data': bus_data_attack2,
            'attack_type': 'load_manipulation',
            'description': '50% load increase at bus 9'
        }
        
        # Scenario 3: Generator setpoint attack
        gen_data_attack3 = self.generator_data.copy()
        gen_data_attack3.loc[gen_data_attack3['bus_id'] == 2, 'voltage_setpoint'] = 0.95
        scenarios['generator_setpoint_attack'] = {
            **base_data,
            'generator_data': gen_data_attack3,
            'attack_type': 'generator_setpoint',
            'description': 'Voltage setpoint attack on generator at bus 2'
        }
        
        # Scenario 4: Multiple line impedance attack
        line_data_attack4 = self.line_data.copy()
        line_data_attack4.loc[0, 'reactance'] *= 1.2  # Line 1-2
        line_data_attack4.loc[1, 'reactance'] *= 0.8  # Line 1-5
        scenarios['impedance_manipulation'] = {
            **base_data,
            'line_data': line_data_attack4,
            'attack_type': 'impedance_manipulation',
            'description': 'Line impedance manipulation attack'
        }
        
        return scenarios


class IEEE39Bus:
    """
    IEEE 39-bus (New England) test system implementation
    """
    
    def __init__(self):
        self.name = "IEEE 39-Bus Test System (New England)"
        self.base_mva = 100.0
        self.bus_data = None
        self.line_data = None
        self.generator_data = None
        self.load_data = None
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the IEEE 39-bus system data"""
        
        # Simplified bus data for the 39-bus system
        # In practice, this would be loaded from standard test case files
        bus_ids = list(range(1, 40))
        n_buses = len(bus_ids)
        
        # Create basic bus data structure
        bus_data_array = []
        for i, bus_id in enumerate(bus_ids):
            if bus_id == 31:  # Slack bus
                bus_type = 3
                v_mag = 1.0
            elif bus_id in [30, 32, 33, 34, 35, 36, 37, 38, 39]:  # Generator buses
                bus_type = 2
                v_mag = 1.0
            else:  # Load buses
                bus_type = 1
                v_mag = 1.0
            
            # Sample load values (would be actual values in real implementation)
            p_load = np.random.uniform(50, 300) if bus_type == 1 else 0
            q_load = p_load * 0.3 if bus_type == 1 else 0
            
            bus_data_array.append([
                bus_id, bus_type, v_mag, 0.0, p_load, q_load, 0.0, 0.0, 345
            ])
        
        self.bus_data = pd.DataFrame(bus_data_array, columns=[
            'bus_id', 'type', 'voltage_magnitude', 'voltage_angle',
            'p_load', 'q_load', 'p_gen', 'q_gen', 'base_kv'
        ])
        
        # Create sample line data (simplified)
        line_data_array = []
        line_id = 0
        for i in range(1, 40):
            # Create a connected network
            if i < 39:
                line_data_array.append([
                    i, i+1, 
                    np.random.uniform(0.01, 0.05),  # resistance
                    np.random.uniform(0.05, 0.2),   # reactance
                    np.random.uniform(0.01, 0.05),  # susceptance
                    1.0, 0.0
                ])
            
            # Add some additional connections for a more realistic topology
            if i <= 35 and i % 3 == 0:
                line_data_array.append([
                    i, i+3,
                    np.random.uniform(0.02, 0.08),
                    np.random.uniform(0.1, 0.3),
                    np.random.uniform(0.02, 0.08),
                    1.0, 0.0
                ])
        
        self.line_data = pd.DataFrame(line_data_array, columns=[
            'from_bus', 'to_bus', 'resistance', 'reactance',
            'susceptance', 'tap_ratio', 'phase_shift'
        ])
        
        # Generator data
        gen_buses = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        generator_data_array = []
        for bus_id in gen_buses:
            generator_data_array.append([
                bus_id, 
                np.random.uniform(500, 1000),  # p_max
                0.0,                           # p_min
                np.random.uniform(200, 500),   # q_max
                np.random.uniform(-200, -100), # q_min
                1.0,                           # voltage_setpoint
                100.0                          # mbase
            ])
        
        self.generator_data = pd.DataFrame(generator_data_array, columns=[
            'bus_id', 'p_max', 'p_min', 'q_max', 'q_min', 'voltage_setpoint', 'mbase'
        ])
        
        # Load data
        load_buses = self.bus_data[self.bus_data['p_load'] > 0]
        self.load_data = load_buses[['bus_id', 'p_load', 'q_load']].copy()
    
    def get_system_data(self) -> Dict[str, Any]:
        """Get complete system data"""
        return {
            'name': self.name,
            'base_mva': self.base_mva,
            'bus_data': self.bus_data,
            'line_data': self.line_data,
            'generator_data': self.generator_data,
            'load_data': self.load_data
        }
    
    def create_attack_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Create attack scenarios for the 39-bus system"""
        base_data = self.get_system_data()
        scenarios = {}
        
        # Large-scale load attack
        bus_data_attack1 = self.bus_data.copy()
        load_buses = [3, 7, 15, 20, 25]
        for bus_id in load_buses:
            mask = bus_data_attack1['bus_id'] == bus_id
            bus_data_attack1.loc[mask, 'p_load'] *= 1.3
            bus_data_attack1.loc[mask, 'q_load'] *= 1.3
        
        scenarios['coordinated_load_attack'] = {
            **base_data,
            'bus_data': bus_data_attack1,
            'attack_type': 'coordinated_load_attack',
            'description': 'Coordinated 30% load increase at multiple buses'
        }
        
        # Generator coordination attack
        gen_data_attack2 = self.generator_data.copy()
        gen_data_attack2.loc[gen_data_attack2['bus_id'].isin([32, 34, 36]), 'voltage_setpoint'] = 0.92
        
        scenarios['generator_coordination_attack'] = {
            **base_data,
            'generator_data': gen_data_attack2,
            'attack_type': 'generator_coordination_attack',
            'description': 'Coordinated voltage setpoint reduction at multiple generators'
        }
        
        return scenarios


class IEEE118Bus:
    """
    IEEE 118-bus test system implementation
    """
    
    def __init__(self):
        self.name = "IEEE 118-Bus Test System"
        self.base_mva = 100.0
        self.bus_data = None
        self.line_data = None
        self.generator_data = None
        self.load_data = None
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the IEEE 118-bus system data"""
        
        # Create large-scale test system data
        bus_ids = list(range(1, 119))
        n_buses = len(bus_ids)
        
        # Bus data with realistic distribution
        bus_data_array = []
        generator_buses = [1, 4, 6, 8, 10, 12, 15, 18, 19, 24, 25, 26, 27, 31, 32, 
                          34, 36, 40, 42, 46, 49, 54, 55, 56, 59, 61, 62, 65, 66, 
                          69, 70, 72, 73, 74, 76, 77, 80, 85, 87, 89, 90, 91, 92, 
                          99, 100, 103, 104, 105, 107, 110, 111, 112, 113, 116]
        
        slack_bus = 69  # Typical slack bus for IEEE 118
        
        for bus_id in bus_ids:
            if bus_id == slack_bus:
                bus_type = 3
                v_mag = 1.035
            elif bus_id in generator_buses:
                bus_type = 2
                v_mag = np.random.uniform(1.0, 1.05)
            else:
                bus_type = 1
                v_mag = np.random.uniform(0.95, 1.05)
            
            # Load distribution
            if bus_type == 1:  # Load bus
                p_load = np.random.uniform(10, 100)
                q_load = p_load * np.random.uniform(0.2, 0.4)
            else:
                p_load = 0.0
                q_load = 0.0
            
            bus_data_array.append([
                bus_id, bus_type, v_mag, 0.0, p_load, q_load, 0.0, 0.0, 
                np.random.choice([138, 230, 345])  # Voltage levels
            ])
        
        self.bus_data = pd.DataFrame(bus_data_array, columns=[
            'bus_id', 'type', 'voltage_magnitude', 'voltage_angle',
            'p_load', 'q_load', 'p_gen', 'q_gen', 'base_kv'
        ])
        
        # Create realistic transmission network topology
        line_data_array = []
        
        # Create backbone transmission lines
        for i in range(1, 118):
            if i % 2 == 0:  # Even buses connect to next even bus
                if i + 2 <= 118:
                    line_data_array.append([
                        i, i+2,
                        np.random.uniform(0.005, 0.03),
                        np.random.uniform(0.05, 0.15),
                        np.random.uniform(0.01, 0.05),
                        1.0, 0.0
                    ])
            else:  # Odd buses connect to next odd bus
                if i + 2 <= 118:
                    line_data_array.append([
                        i, i+2,
                        np.random.uniform(0.005, 0.03),
                        np.random.uniform(0.05, 0.15),
                        np.random.uniform(0.01, 0.05),
                        1.0, 0.0
                    ])
        
        # Add cross-connections for realistic topology
        for i in range(1, 115, 5):
            line_data_array.append([
                i, i+3,
                np.random.uniform(0.01, 0.05),
                np.random.uniform(0.1, 0.25),
                np.random.uniform(0.02, 0.08),
                1.0, 0.0
            ])
        
        # Additional meshed connections
        np.random.seed(42)  # For reproducibility
        for _ in range(50):  # Add 50 random connections
            from_bus = np.random.randint(1, 119)
            to_bus = np.random.randint(1, 119)
            if from_bus != to_bus:
                line_data_array.append([
                    from_bus, to_bus,
                    np.random.uniform(0.01, 0.08),
                    np.random.uniform(0.05, 0.3),
                    np.random.uniform(0.01, 0.1),
                    1.0, 0.0
                ])
        
        self.line_data = pd.DataFrame(line_data_array, columns=[
            'from_bus', 'to_bus', 'resistance', 'reactance',
            'susceptance', 'tap_ratio', 'phase_shift'
        ])
        
        # Generator data
        generator_data_array = []
        for bus_id in generator_buses:
            generator_data_array.append([
                bus_id,
                np.random.uniform(100, 800),   # p_max
                0.0,                           # p_min
                np.random.uniform(50, 300),    # q_max
                np.random.uniform(-100, -30),  # q_min
                np.random.uniform(1.0, 1.05),  # voltage_setpoint
                100.0                          # mbase
            ])
        
        self.generator_data = pd.DataFrame(generator_data_array, columns=[
            'bus_id', 'p_max', 'p_min', 'q_max', 'q_min', 'voltage_setpoint', 'mbase'
        ])
        
        # Load data
        load_buses = self.bus_data[self.bus_data['p_load'] > 0]
        self.load_data = load_buses[['bus_id', 'p_load', 'q_load']].copy()
    
    def get_system_data(self) -> Dict[str, Any]:
        """Get complete system data"""
        return {
            'name': self.name,
            'base_mva': self.base_mva,
            'bus_data': self.bus_data,
            'line_data': self.line_data,
            'generator_data': self.generator_data,
            'load_data': self.load_data
        }
    
    def create_attack_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Create large-scale attack scenarios for the 118-bus system"""
        base_data = self.get_system_data()
        scenarios = {}
        
        # Large-scale coordinated attack
        bus_data_attack1 = self.bus_data.copy()
        attack_buses = list(range(10, 30)) + list(range(50, 70)) + list(range(90, 110))
        
        for bus_id in attack_buses:
            mask = bus_data_attack1['bus_id'] == bus_id
            if bus_data_attack1.loc[mask, 'p_load'].iloc[0] > 0:
                bus_data_attack1.loc[mask, 'p_load'] *= 1.4
                bus_data_attack1.loc[mask, 'q_load'] *= 1.4
        
        scenarios['large_scale_coordinated_attack'] = {
            **base_data,
            'bus_data': bus_data_attack1,
            'attack_type': 'large_scale_coordinated_attack',
            'description': 'Large-scale coordinated 40% load increase at 60 buses'
        }
        
        # Transmission line cascade attack
        line_data_attack2 = self.line_data.copy()
        # Remove several critical transmission lines
        lines_to_remove = [(1, 3), (10, 12), (25, 27), (50, 52), (75, 77), (100, 102)]
        
        for from_bus, to_bus in lines_to_remove:
            mask = ((line_data_attack2['from_bus'] == from_bus) & 
                   (line_data_attack2['to_bus'] == to_bus)) | \
                   ((line_data_attack2['from_bus'] == to_bus) & 
                   (line_data_attack2['to_bus'] == from_bus))
            line_data_attack2 = line_data_attack2[~mask]
        
        scenarios['transmission_cascade_attack'] = {
            **base_data,
            'line_data': line_data_attack2,
            'attack_type': 'transmission_cascade_attack',
            'description': 'Coordinated transmission line outage attack'
        }
        
        return scenarios


def get_ieee_test_system(system_name: str):
    """
    Factory function to get IEEE test system
    
    Args:
        system_name: Name of the system ('14bus', '39bus', '118bus')
        
    Returns:
        IEEE test system object
    """
    systems = {
        '14bus': IEEE14Bus,
        '39bus': IEEE39Bus,
        '118bus': IEEE118Bus
    }
    
    if system_name.lower() not in systems:
        raise ValueError(f"Unknown system: {system_name}. Available: {list(systems.keys())}")
    
    return systems[system_name.lower()]()