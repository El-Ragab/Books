#!/usr/bin/env python3
"""
Test script for the Cyber-Attack Detection Framework
Quick validation of core functionality
"""

import sys
import os
sys.path.append('/workspace/src')

import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_generation():
    """Test data generation functionality"""
    logger.info("Testing data generation...")
    
    from data_generation import PowerSystemDataGenerator
    
    # Initialize generator
    generator = PowerSystemDataGenerator('14bus')
    
    # Generate small dataset for testing
    normal_data = generator.generate_normal_operation_data(duration_hours=0.1, noise_level=0.05)
    attack_data = generator.generate_attack_data('load_manipulation_bus9', duration_hours=0.1)
    
    # Create ML dataset
    ml_dataset = generator.create_ml_dataset(
        normal_hours=0.5,
        attack_scenarios=['load_manipulation_bus9'],
        attack_hours=0.1,
        window_size=20
    )
    
    logger.info(f"✓ Data generation successful")
    logger.info(f"  Normal data: {len(normal_data['timestamps'])} samples")
    logger.info(f"  Attack data: {len(attack_data['timestamps'])} samples") 
    logger.info(f"  ML dataset: {ml_dataset['features'].shape}")
    
    return True

def test_ml_algorithms():
    """Test machine learning algorithms"""
    logger.info("Testing ML algorithms...")
    
    from algorithms.ml_classifiers import RandomForestDetector
    from algorithms.unsupervised_learning import IsolationForestDetector
    
    # Create simple test data
    np.random.seed(42)
    X_normal = np.random.normal(0, 1, (100, 10))
    X_attack = np.random.normal(2, 1, (50, 10))
    X = np.vstack([X_normal, X_attack])
    y = np.hstack([np.zeros(100), np.ones(50)])
    
    # Test Random Forest
    rf = RandomForestDetector()
    rf_results = rf.train(X, y, optimize_hyperparams=False)
    rf_eval = rf.evaluate(X, y)
    
    # Test Isolation Forest
    iso = IsolationForestDetector()
    iso_results = iso.train(X_normal)
    iso_eval = iso.evaluate(X, y)
    
    logger.info(f"✓ ML algorithms successful")
    logger.info(f"  Random Forest accuracy: {rf_eval['accuracy']:.3f}")
    logger.info(f"  Isolation Forest accuracy: {iso_eval['accuracy']:.3f}")
    
    return True

def test_power_system_analysis():
    """Test power system analysis"""
    logger.info("Testing power system analysis...")
    
    from power_system.ieee_test_systems import get_ieee_test_system
    from power_system.y_matrix_analysis import YMatrixAnalyzer
    
    # Get test system
    test_system = get_ieee_test_system('14bus')
    system_data = test_system.get_system_data()
    
    # Test Y-matrix analysis
    y_analyzer = YMatrixAnalyzer()
    baseline_results = y_analyzer.establish_baseline(
        system_data['bus_data'], 
        system_data['line_data']
    )
    
    # Test attack scenario
    attack_scenarios = test_system.create_attack_scenarios()
    line_outage = attack_scenarios['line_outage_1_2']
    
    attack_results = y_analyzer.detect_topology_attacks(
        line_outage['bus_data'],
        line_outage['line_data']
    )
    
    logger.info(f"✓ Power system analysis successful")
    logger.info(f"  System: {system_data['name']}")
    logger.info(f"  Buses: {len(system_data['bus_data'])}")
    logger.info(f"  Attack detected: {attack_results['is_attack_detected']}")
    
    return True

def test_statistical_methods():
    """Test statistical methods"""
    logger.info("Testing statistical methods...")
    
    from algorithms.statistical_methods import ZScoreAnomalyDetector, StatisticalBaseline
    
    # Create test data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (200, 5))
    test_data = np.vstack([
        np.random.normal(0, 1, (150, 5)),  # Normal
        np.random.normal(3, 1, (50, 5))    # Anomalous
    ])
    
    # Test Z-score detector
    zscore = ZScoreAnomalyDetector()
    zscore_results = zscore.train(normal_data)
    anomalies, _, _ = zscore.detect_anomalies(test_data)
    
    # Test statistical baseline
    baseline = StatisticalBaseline()
    baseline_results = baseline.establish_baseline(normal_data)
    
    logger.info(f"✓ Statistical methods successful")
    logger.info(f"  Z-score anomaly rate: {np.mean(anomalies):.3f}")
    logger.info(f"  Baseline established for {normal_data.shape[1]} features")
    
    return True

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("CYBER-ATTACK DETECTION FRAMEWORK - QUICK TEST")
    logger.info("="*60)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("ML Algorithms", test_ml_algorithms), 
        ("Power System Analysis", test_power_system_analysis),
        ("Statistical Methods", test_statistical_methods)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- {test_name} ---")
            success = test_func()
            if success:
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} FAILED: {str(e)}")
    
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Tests Passed: {passed}")
    logger.info(f"Tests Failed: {failed}")
    logger.info(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Framework is ready for use.")
        return True
    else:
        logger.error(f"❌ {failed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)