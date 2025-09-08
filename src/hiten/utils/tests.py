"""Test discovery and execution utilities for the hiten package.

This module provides functions for automatically discovering and running
test files throughout the project using pytest. It includes utilities
for finding test files, filtering tests by name patterns, and executing
them with appropriate configuration.

Notes
-----
This module is designed to work with pytest and automatically discovers
test files matching the pattern test_*.py throughout the project.
"""

import pytest
import sys
import os
import glob
from pathlib import Path


def find_test_files() -> list[str]:
    """Find all test_*.py files in the project.
    
    This function recursively searches through the project directory
    structure to locate all Python files that match the test_*.py pattern.
    It also ensures that the src directory is added to the Python path
    so that imports work correctly during test execution.
    
    Returns
    -------
    list of str
        List of absolute file paths to all discovered test files.
        
    Notes
    -----
    The function automatically adds the src directory to sys.path
    to ensure proper import resolution during test execution.
    
    Examples
    --------
    >>> test_files = find_test_files()
    >>> len(test_files) > 0
    True
    >>> all(f.endswith('.py') for f in test_files)
    True
    """
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

def main() -> None:
    """Run the tests using pytest.
    
    This function discovers test files, optionally filters them based on
    command-line arguments, and executes them using pytest with appropriate
    configuration options.
    
    Parameters
    ----------
    sys.argv[1:] : list of str
        Command-line arguments for filtering tests. If no arguments are
        provided, all tests are run. If arguments are provided, only tests
        containing those strings in their file paths are executed.
        
    Notes
    -----
    The function uses pytest with the following default options:
    - -xv: Verbose output with extra information
    - -s: Don't capture output (allows print statements to show)
    
    Examples
    --------
    Run all tests:
        python tests.py
        
    Run only polynomial-related tests:
        python tests.py polynomials
        
    Run multiple test categories:
        python tests.py polynomials linalg
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
