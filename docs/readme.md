# HITEN Documentation

This directory contains the Sphinx documentation for the HITEN package.

## Building the Documentation

### Prerequisites

First, install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

### Building HTML Documentation

To build the HTML documentation:

```bash
# Using make (Linux/macOS)
make html

# Using make.bat (Windows)
make.bat html

# Using sphinx-build directly
sphinx-build -b html . _build/html
```

The built documentation will be available in `_build/html/index.html`.

### Live Development

For live development with automatic rebuilding:

```bash
# Install sphinx-autobuild
pip install sphinx-autobuild

# Start live server
make livehtml
# or on Windows: make.bat livehtml
```

This will start a local server (usually at http://127.0.0.1:8000) that automatically rebuilds the documentation when files change.

### Building Other Formats

```bash
# PDF documentation
make latexpdf

# EPUB documentation
make epub
```

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start guide
- `api/` - API reference documentation
- `user_guide/` - User guide documentation
- `examples/` - Example documentation
- `developer/` - Developer documentation
- `_static/` - Static files (CSS, images, etc.)
- `_templates/` - Custom templates

## Adding New Documentation

1. **API Documentation**: Add new modules to the appropriate file in `api/`
2. **User Guide**: Add new guides to `user_guide/`
3. **Examples**: Add new examples to `examples/`
4. **Developer Docs**: Add developer information to `developer/`

## Configuration

The Sphinx configuration is in `conf.py`. Key settings:

- **Extensions**: Configured for autodoc, napoleon, math support, etc.
- **Theme**: Uses sphinx-rtd-theme
- **Autodoc**: Automatically generates API docs from docstrings
- **Math**: Supports LaTeX math notation
- **Cross-references**: Configured for Python, NumPy, SciPy, etc.

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the package is installed in development mode
2. **Missing dependencies**: Install all documentation dependencies
3. **Build errors**: Check for syntax errors in RST files
4. **Missing modules**: Ensure all modules are properly imported in `conf.py`

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- See the [sphinx-rtd-theme documentation](https://sphinx-rtd-theme.readthedocs.io/)
- Check the [GitHub Issues](https://github.com/iamgadmarconi/hiten/issues)

## Contributing

When contributing to the documentation:

1. Follow the existing style and structure
2. Use proper RST syntax
3. Include cross-references where appropriate
4. Test the build locally before submitting
5. Update the table of contents if adding new sections
