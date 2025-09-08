#!/usr/bin/env python3
"""
Script to detect potential missing Sphinx references in docstrings.

This script searches for common patterns that suggest missing Sphinx cross-references
in docstrings, such as class names, function names, or module names that should be
wrapped in Sphinx roles but aren't.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple, Dict

def find_potential_missing_refs(file_path: str) -> List[Tuple[int, str, str]]:
    """
    Find potential missing Sphinx references in a Python file.
    
    Returns:
        List of tuples: (line_number, line_content, suggestion)
    """
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Patterns to detect potential missing references
    patterns = [
        # Class names in docstrings (PascalCase followed by lowercase)
        (r'([A-Z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*)\s+[a-z]', 'class'),
        # Function names in docstrings (snake_case or camelCase)
        (r'([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+)\s*\(', 'func'),
        # Module names (often all lowercase with underscores)
        (r'([a-z][a-zA-Z0-9_]*\.[a-z][a-zA-Z0-9_]*)', 'mod'),
        # Method names (snake_case)
        (r'([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+)\s*\(', 'meth'),
        # Attribute names (often snake_case)
        (r'([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+)\s*[=:]', 'attr'),
    ]
    
    in_docstring = False
    docstring_quote = None
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Track docstring state
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
                docstring_quote = '"""' if stripped.startswith('"""') else "'''"
            elif stripped.endswith(docstring_quote) and len(stripped) > 3:
                in_docstring = False
                docstring_quote = None
        
        # Only check inside docstrings
        if not in_docstring:
            continue
            
        # Skip lines that already have Sphinx roles
        if ':' in line and any(role in line for role in [':class:', ':func:', ':meth:', ':mod:', ':attr:']):
            continue
            
        # Check for potential missing references
        for pattern, role_type in patterns:
            matches = re.finditer(pattern, line)
            for match in matches:
                potential_ref = match.group(1)
                
                # Skip common false positives
                if any(skip in potential_ref.lower() for skip in [
                    'numpy', 'scipy', 'matplotlib', 'numba', 'dataclasses',
                    'typing', 'abc', 'collections', 'itertools', 'functools',
                    'warnings', 'logging', 'pathlib', 'os', 'sys', 're',
                    'math', 'random', 'json', 'pickle', 'copy', 'deepcopy',
                    'defaultdict', 'namedtuple', 'deque', 'counter', 'ordereddict',
                    'callable', 'optional', 'union', 'list', 'dict', 'tuple',
                    'set', 'frozenset', 'bytes', 'bytearray', 'str', 'int',
                    'float', 'bool', 'complex', 'object', 'type', 'none',
                    'true', 'false', 'ellipsis', 'notimplemented', 'self',
                    'cls', 'super', 'staticmethod', 'classmethod', 'property',
                    'abstractmethod', 'abstractproperty', 'overload', 'final',
                    'runtime', 'value', 'type', 'attribute', 'key', 'index',
                    'stop', 'iteration', 'generator', 'yield', 'return',
                    'raise', 'except', 'try', 'finally', 'with', 'as',
                    'import', 'from', 'def', 'class', 'if', 'else', 'elif',
                    'for', 'while', 'break', 'continue', 'pass', 'del',
                    'global', 'nonlocal', 'lambda', 'and', 'or', 'not',
                    'in', 'is', 'assert', 'async', 'await', 'yield',
                    'match', 'case', 'if', 'else', 'elif', 'for', 'while',
                    'break', 'continue', 'pass', 'del', 'global', 'nonlocal',
                    'lambda', 'and', 'or', 'not', 'in', 'is', 'assert',
                    'async', 'await', 'yield', 'match', 'case'
                ]):
                    continue
                
                # Skip if it's already a Sphinx reference
                if f':{role_type}:`' in line:
                    continue
                
                # Skip if it's in a comment or code block
                if line.strip().startswith('#') or line.strip().startswith('>>>'):
                    continue
                
                # Skip if it's a variable assignment or function call
                if '=' in line and potential_ref in line.split('=')[0]:
                    continue
                
                # Skip if it's a type hint
                if ':' in line and potential_ref in line.split(':')[0]:
                    continue
                
                # Skip if it's a string literal
                if potential_ref in line and ('"' in line or "'" in line):
                    continue
                
                # Skip if it's a number or constant
                if potential_ref.isdigit() or potential_ref.isupper():
                    continue
                
                # Skip if it's a common word
                if potential_ref.lower() in [
                    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
                    'during', 'before', 'after', 'above', 'below', 'between',
                    'among', 'under', 'over', 'around', 'near', 'far', 'here',
                    'there', 'where', 'when', 'why', 'how', 'what', 'who',
                    'which', 'that', 'this', 'these', 'those', 'a', 'an',
                    'some', 'any', 'all', 'both', 'each', 'every', 'either',
                    'neither', 'one', 'two', 'three', 'first', 'second',
                    'last', 'next', 'previous', 'other', 'another', 'same',
                    'different', 'similar', 'different', 'same', 'similar',
                    'different', 'same', 'similar', 'different', 'same'
                ]:
                    continue
                
                # This looks like a potential missing reference
                suggestion = f"Consider using :{role_type}:`{potential_ref}`"
                issues.append((i, line.rstrip(), suggestion))
    
    return issues

def scan_directory(directory: str) -> Dict[str, List[Tuple[int, str, str]]]:
    """
    Scan a directory for potential missing Sphinx references.
    
    Returns:
        Dictionary mapping file paths to lists of issues
    """
    results = {}
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                issues = find_potential_missing_refs(file_path)
                if issues:
                    results[file_path] = issues
    
    return results

def main():
    """Main function to run the detection script."""
    # Scan the src directory
    src_dir = "src"
    if not os.path.exists(src_dir):
        print(f"Directory {src_dir} not found!")
        return
    
    print("Scanning for potential missing Sphinx references...")
    print("=" * 60)
    
    results = scan_directory(src_dir)
    
    if not results:
        print("No potential missing Sphinx references found!")
        return
    
    total_issues = sum(len(issues) for issues in results.values())
    print(f"Found {total_issues} potential missing Sphinx references in {len(results)} files:")
    print()
    
    for file_path, issues in results.items():
        print(f"File: {file_path}")
        print("-" * len(file_path))
        for line_num, line_content, suggestion in issues:
            print(f"  Line {line_num}: {line_content}")
            print(f"  Suggestion: {suggestion}")
            print()
        print()

if __name__ == "__main__":
    main()
