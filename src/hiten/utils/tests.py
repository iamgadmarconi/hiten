import pytest
import sys
import os
import glob
from pathlib import Path


def find_test_files():
    """Find all test_*.py files in the project."""
    # Get the src directory (go up from utils to src)
    src_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    project_root = src_dir.parent
    
    # Add src directory to Python path so imports work correctly
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    test_files = []
    
    # Walk the directory structure to find all test_*.py files
    for root, _, _ in os.walk(src_dir):
        root_path = Path(root)
        # Find all files matching test_*.py pattern
        matches = list(root_path.glob("test_*.py"))
        for match in matches:
            # Convert to string and make path relative to current directory
            test_files.append(str(match.resolve()))
    
    return test_files

def main():
    """Run the tests using pytest.
    
    Usage:
        python main.py               # Run all tests
        python main.py polynomials   # Run only tests containing 'polynomials' in path
        python main.py linalg        # Run only tests containing 'linalg' in path
    """
    args = sys.argv[1:]
    
    # Find all test files
    all_test_files = find_test_files()
    test_paths = []
    
    if not args:
        # Run all tests
        test_paths = all_test_files
    else:
        # Filter tests based on arguments
        for arg in args:
            matching_files = [f for f in all_test_files if arg.lower() in f.lower()]
            test_paths.extend(matching_files)
    
    if not test_paths:
        print("No test files found matching the criteria")
        return
    
    # Print test files being run
    print(f"Running {len(test_paths)} test files:")
    for path in test_paths:
        print(f"  - {os.path.relpath(path)}")
    
    # Add pytest options
    pytest_args = [
        "-xv",
        "-s",
        # "--log-cli-level=INFO",
        # "--log-cli-format=%(asctime)s | %(levelname)7s | %(name)s: %(message)s",
        # "--log-cli-date-format=%Y-%m-%d %H:%M:%S"
    ] + test_paths
    
    # Run the tests
    pytest.main(pytest_args)


if __name__ == "__main__":
    main()
