#!/usr/bin/env python3
"""
Simple structure validation script for the Cyber-Attack Detection Framework
Validates project structure and core files without requiring external dependencies
"""

import os
import sys
from pathlib import Path

def validate_file_structure():
    """Validate that all required files and directories exist"""
    print("Validating project structure...")
    
    required_structure = {
        'src/': ['algorithms/', 'power_system/', 'attacks/', 'protection/', 'utils/'],
        'src/algorithms/': ['__init__.py', 'ml_classifiers.py', 'unsupervised_learning.py', 'statistical_methods.py'],
        'src/power_system/': ['__init__.py', 'jacobian_analysis.py', 'y_matrix_analysis.py', 'ieee_test_systems.py'],
        'data/': ['raw/', 'processed/', 'ieee_systems/'],
        'results/': ['phase1/', 'phase2/', 'phase3/'],
        'notebooks/': ['demo_simulation.ipynb'],
        '.': ['requirements.txt', 'README.md', 'FRAMEWORK_SUMMARY.md', 'src/main.py', 'src/data_generation.py']
    }
    
    missing_items = []
    existing_items = []
    
    for directory, files in required_structure.items():
        dir_path = Path(directory)
        
        if not dir_path.exists():
            missing_items.append(f"Directory: {directory}")
            continue
            
        for file_name in files:
            file_path = dir_path / file_name
            if file_path.exists():
                existing_items.append(str(file_path))
            else:
                missing_items.append(f"File: {file_path}")
    
    print(f"✓ Found {len(existing_items)} required items")
    
    if missing_items:
        print(f"✗ Missing {len(missing_items)} items:")
        for item in missing_items[:10]:  # Show first 10
            print(f"  - {item}")
        if len(missing_items) > 10:
            print(f"  ... and {len(missing_items) - 10} more")
        return False
    else:
        print("✓ All required files and directories exist")
        return True

def validate_python_syntax():
    """Validate Python syntax of core files"""
    print("\nValidating Python syntax...")
    
    python_files = [
        'src/main.py',
        'src/data_generation.py',
        'src/algorithms/__init__.py',
        'src/algorithms/ml_classifiers.py',
        'src/algorithms/unsupervised_learning.py',
        'src/algorithms/statistical_methods.py',
        'src/power_system/__init__.py',
        'src/power_system/jacobian_analysis.py',
        'src/power_system/y_matrix_analysis.py',
        'src/power_system/ieee_test_systems.py'
    ]
    
    syntax_errors = []
    valid_files = []
    
    for file_path in python_files:
        if not Path(file_path).exists():
            continue
            
        try:
            with open(file_path, 'r') as f:
                source = f.read()
            
            compile(source, file_path, 'exec')
            valid_files.append(file_path)
            
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {e}")
    
    print(f"✓ {len(valid_files)} files have valid Python syntax")
    
    if syntax_errors:
        print(f"✗ {len(syntax_errors)} files have syntax errors:")
        for error in syntax_errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ All Python files have valid syntax")
        return True

def validate_documentation():
    """Validate documentation files"""
    print("\nValidating documentation...")
    
    doc_files = {
        'README.md': 'Project documentation',
        'FRAMEWORK_SUMMARY.md': 'Framework summary',
        'requirements.txt': 'Dependencies list'
    }
    
    doc_issues = []
    valid_docs = []
    
    for doc_file, description in doc_files.items():
        if not Path(doc_file).exists():
            doc_issues.append(f"Missing {description}: {doc_file}")
            continue
        
        try:
            with open(doc_file, 'r') as f:
                content = f.read()
            
            if len(content.strip()) < 100:  # Minimum content check
                doc_issues.append(f"Insufficient content in {doc_file}")
            else:
                valid_docs.append(doc_file)
                
        except Exception as e:
            doc_issues.append(f"Error reading {doc_file}: {e}")
    
    print(f"✓ {len(valid_docs)} documentation files are valid")
    
    if doc_issues:
        print(f"✗ Documentation issues:")
        for issue in doc_issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All documentation files are valid")
        return True

def count_code_lines():
    """Count lines of code in the project"""
    print("\nCounting lines of code...")
    
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    total_lines = 0
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                total_lines += 1
                stripped = line.strip()
                
                if not stripped:
                    blank_lines += 1
                elif stripped.startswith('#'):
                    comment_lines += 1
                elif '"""' in stripped or "'''" in stripped:
                    comment_lines += 1
                else:
                    code_lines += 1
                    
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    print(f"✓ Code statistics:")
    print(f"  - Python files: {len(python_files)}")
    print(f"  - Total lines: {total_lines:,}")
    print(f"  - Code lines: {code_lines:,}")
    print(f"  - Comment/doc lines: {comment_lines:,}")
    print(f"  - Blank lines: {blank_lines:,}")
    
    return True

def main():
    """Run all validation checks"""
    print("="*60)
    print("CYBER-ATTACK DETECTION FRAMEWORK - STRUCTURE VALIDATION")
    print("="*60)
    
    checks = [
        ("File Structure", validate_file_structure),
        ("Python Syntax", validate_python_syntax),
        ("Documentation", validate_documentation),
        ("Code Statistics", count_code_lines)
    ]
    
    passed = 0
    failed = 0
    
    for check_name, check_func in checks:
        try:
            print(f"\n--- {check_name} ---")
            success = check_func()
            if success:
                passed += 1
                print(f"✓ {check_name} PASSED")
            else:
                failed += 1
                print(f"✗ {check_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {check_name} FAILED: {str(e)}")
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Checks Passed: {passed}")
    print(f"Checks Failed: {failed}")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 FRAMEWORK STRUCTURE IS VALID!")
        print("✓ All files and directories are in place")
        print("✓ Python syntax is correct")
        print("✓ Documentation is complete")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full test: python test_framework.py")
        print("3. Try the demo: jupyter notebook notebooks/demo_simulation.ipynb")
        return True
    else:
        print(f"\n❌ {failed} validation check(s) failed.")
        print("Please address the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)