# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Kea'
copyright = '2026, Mark Bishop, Yvette Perrott, Tulasi Parashar, Sean Oughton'
author = 'Mark Bishop, Yvette Perrott, Tulasi Parashar, Sean Oughton'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Automatically pulls docstrings from your code
    "sphinx.ext.napoleon", # Supports NumPy/Google style docstrings
    "sphinx.ext.mathjax",  # Renders LaTeX math via MathJax
    "myst_parser",         # Enables Markdown support
]
# Enable dollar signs ($ and $$) for inline and block math in Markdown
myst_enable_extensions = ["dollarmath", "amsmath"]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # The classic, clean Read the Docs look
html_static_path = ['_static']
