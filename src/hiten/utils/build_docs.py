"""Build script for documentation.

This script provides a convenient way to build the documentation
with various options and configurations.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"OK - {cmd}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"ERROR - {cmd}")
        print(f"Error: {e.stderr}")
        return None


def install_docs_deps():
    """Install documentation dependencies."""
    print("Installing documentation dependencies...")
    cmd = "pip install -e .[docs]"
    return run_command(cmd)


def build_html(clean=False, live=False):
    """Build HTML documentation."""
    # Get the project root directory (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent
    docs_dir = project_root / "docs"
    
    if not docs_dir.exists():
        print(f"Error: Documentation directory {docs_dir} does not exist!")
        return False
    
    if clean:
        print("Cleaning previous builds...")
        # Use Windows batch file on Windows, make on Unix
        if os.name == 'nt':
            run_command("make.bat clean", cwd=str(docs_dir))
        else:
            run_command("make clean", cwd=str(docs_dir))
    
    if live:
        print("Starting live HTML build...")
        if os.name == 'nt':
            cmd = "make.bat livehtml"
        else:
            cmd = "make livehtml"
    else:
        print("Building HTML documentation...")
        if os.name == 'nt':
            cmd = "make.bat html"
        else:
            cmd = "make html"
    
    return run_command(cmd, cwd=str(docs_dir))


def build_pdf():
    """Build PDF documentation."""
    print("Building PDF documentation...")
    project_root = Path(__file__).parent.parent.parent.parent
    docs_dir = project_root / "docs"
    if os.name == 'nt':
        return run_command("make.bat latexpdf", cwd=str(docs_dir))
    else:
        return run_command("make latexpdf", cwd=str(docs_dir))


def build_epub():
    """Build EPUB documentation."""
    print("Building EPUB documentation...")
    project_root = Path(__file__).parent.parent.parent.parent
    docs_dir = project_root / "docs"
    if os.name == 'nt':
        return run_command("make.bat epub", cwd=str(docs_dir))
    else:
        return run_command("make epub", cwd=str(docs_dir))


def check_docs():
    """Check documentation for errors."""
    print("Checking documentation...")
    project_root = Path(__file__).parent.parent.parent.parent
    docs_dir = project_root / "docs"
    
    # Check for broken links
    cmd = "sphinx-build -b linkcheck . _build/linkcheck"
    result = run_command(cmd, cwd=str(docs_dir))
    
    if result:
        print("OK - Link check completed")
    else:
        print("ERROR - Link check failed")
    
    # Check for spelling errors
    cmd = "sphinx-build -b spelling . _build/spelling"
    result = run_command(cmd, cwd=str(docs_dir))
    
    if result:
        print("OK - Spelling check completed")
    else:
        print("ERROR - Spelling check failed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build HITEN documentation")
    parser.add_argument(
        "--format", 
        choices=["html", "pdf", "epub", "all"], 
        default="html",
        help="Documentation format to build"
    )
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Clean previous builds before building"
    )
    parser.add_argument(
        "--live", 
        action="store_true",
        help="Start live HTML build (auto-reload on changes)"
    )
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Install documentation dependencies"
    )
    parser.add_argument(
        "--check", 
        action="store_true",
        help="Check documentation for errors"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_docs_deps():
            print("Failed to install dependencies")
            sys.exit(1)
    
    # Check documentation if requested
    if args.check:
        check_docs()
        return
    
    # Build documentation
    success = True
    
    if args.format == "html":
        success = build_html(clean=args.clean, live=args.live)
    elif args.format == "pdf":
        success = build_pdf()
    elif args.format == "epub":
        success = build_epub()
    elif args.format == "all":
        success = (
            build_html(clean=args.clean) and
            build_pdf() and
            build_epub()
        )
    
    if success:
        print("\nDocumentation build completed successfully!")
        if args.format == "html" and not args.live:
            print("HTML documentation available at: docs/_build/html/index.html")
    else:
        print("\nDocumentation build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
