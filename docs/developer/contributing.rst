Contributing to HITEN
=====================

We welcome contributions to HITEN! This guide will help you get started with contributing to the project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a new branch** for your feature or bugfix
4. **Install in development mode** with dependencies

.. code-block:: bash

   git clone https://github.com/your-username/hiten.git
   cd hiten
   pip install -e ".[dev,docs]"

Development Setup
-----------------

Install all development dependencies:

.. code-block:: bash

   pip install -e ".[dev,docs]"

This installs:
- **pytest** for testing
- **black** for code formatting
- **ruff** for linting
- **sphinx** and related tools for documentation

Code Style
----------

HITEN follows strict coding standards:

- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for all public functions
- **Docstrings** following NumPy style

Run the formatters before committing:

.. code-block:: bash

   black src/
   ruff check src/ --fix

Testing
-------

Write tests for all new functionality:

.. code-block:: bash

   pytest src/hiten/algorithms/your_module/tests/

Documentation
-------------

- Update docstrings for all new functions and classes
- Add examples to the documentation
- Update the user guide if adding new features
- Follow the project's docstring style guide

Pull Request Process
--------------------

1. **Create a feature branch** from main
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run all tests** to ensure nothing is broken
6. **Submit a pull request** with a clear description

Pull Request Guidelines
-----------------------

- **Clear title** describing the change
- **Detailed description** of what was changed and why
- **Reference issues** that are fixed
- **Include tests** for new functionality
- **Update documentation** if needed

Code Review Process
-------------------

All pull requests require review before merging:

- **Automated checks** must pass (tests, linting, formatting)
- **Code review** by maintainers
- **Documentation review** for new features
- **Approval** from at least one maintainer

Issue Reporting
---------------

When reporting issues:

1. **Check existing issues** first
2. **Use the issue template** provided
3. **Include minimal reproduction code**
4. **Specify your environment** (OS, Python version, etc.)
5. **Provide error messages** and stack traces

Feature Requests
----------------

For new features:

1. **Check existing issues** and discussions
2. **Describe the use case** clearly
3. **Explain the expected behavior**
4. **Consider implementation complexity**
5. **Discuss with maintainers** if needed

Development Guidelines
----------------------

- **Keep functions focused** and single-purpose
- **Use descriptive names** for variables and functions
- **Add type hints** for all public APIs
- **Write comprehensive docstrings**
- **Include examples** in docstrings
- **Test edge cases** and error conditions

Mathematical Accuracy
---------------------

Since HITEN is a scientific computing library:

- **Verify mathematical correctness** of all algorithms
- **Include references** to relevant literature
- **Test against known solutions** where possible
- **Document assumptions** and limitations
- **Consider numerical stability**

Performance Considerations
-------------------------

- **Profile code** for performance bottlenecks
- **Use appropriate data structures**
- **Consider memory usage** for large computations
- **Optimize critical paths** in algorithms
- **Use numba** for computationally intensive functions

Questions?
----------

If you have questions about contributing:

- **Check the documentation** first
- **Search existing issues** and discussions
- **Create a new issue** with the "question" label
- **Join the discussions** on GitHub

Thank you for contributing to HITEN! 🚀
