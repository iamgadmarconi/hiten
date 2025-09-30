"""Sphinx configuration for hiten documentation.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Project information
project = "hiten"
copyright = "2025, Gad Marconi"
author = "Gad Marconi"
release = "0.5.0"
version = "0.5.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.extlinks",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "style_nav_header_background": "#1a1a1a",
    "logo_only": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    "**": [
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "hitendoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": "",
    # Latex figure (float) alignment
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        "index",
        "hiten.tex",
        "HITEN Documentation",
        "Gad Marconi",
        "manual",
    ),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        "index",
        "hiten",
        "HITEN Documentation",
        [author],
        1,
    )
]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "hiten",
        "HITEN Documentation",
        author,
        "hiten",
        "Computational Toolkit for the Circular Restricted Three-Body Problem",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -----------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "imported-members": False,
}

# Handle import errors gracefully
autodoc_mock_imports = []

# Don't show type hints in the signature
autodoc_typehints = "description"

# -- Options for autosummary extension -------------------------------------
autosummary_generate = True
autosummary_imported_members = True

# -- Options for napoleon extension ----------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx extension -------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

# -- Options for todo extension --------------------------------------------
todo_include_todos = True

# -- Options for extlinks extension ----------------------------------------
extlinks = {
    "issue": ("https://github.com/iamgadmarconi/hiten/issues/%s", "issue %s"),
    "pr": ("https://github.com/iamgadmarconi/hiten/pull/%s", "PR %s"),
}

# -- Options for MathJax extension ----------------------------------------
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# -- Options for viewcode extension ---------------------------------------
viewcode_follow_imports_missing = True

# -- Options for coverage extension ---------------------------------------
coverage_show_missing_items = True
coverage_ignore_modules = []
coverage_ignore_functions = []
coverage_ignore_classes = []

# -- Options for doctest extension ----------------------------------------
doctest_global_setup = """
import numpy as np
import matplotlib.pyplot as plt
from hiten import *
"""

# -- Options for MyST parser ----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Custom configuration -------------------------------------------------

# Ensure our custom stylesheet is loaded on every page
html_css_files = [
    "custom.css",
]


# -- Options for HTML output ----------------------------------------------
html_title = "HITEN Documentation"
html_short_title = "HITEN"
html_logo = "_static/hiten-cropped.svg"
html_favicon = "_static/hiten.png"

# -- Options for EPUB output ----------------------------------------------
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]
