# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# import os
# import sys

# sys.path.insert(0, os.path.abspath("../../src"))  # Ajusta el path según la ubicación de tu código
# os.chdir("/Users/danielmpr/Documents/ITAM/ArquitecturaGranEscala/ArquitecturaGranEscala/")

project = 'ArquitecturaGranEscalaDoc'
copyright = '2025, DanielMichellPerezRuiz'
author = 'DanielMichellPerezRuiz'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Para documentar automáticamente el código
    "sphinx.ext.napoleon",  # Soporte para docstrings estilo Google/Numpy
    "myst_parser",  # Soporte para Markdown con MyST
]

# Configurar MyST para usar archivos Markdown
myst_enable_extensions = [
    "colon_fence",  # Permite bloques de código con :::
    "deflist",  # Listas de definiciones
    "dollarmath",  # Soporte para ecuaciones matemáticas en $$
    "html_image",  # Soporte para imágenes en Markdown
]

source_suffix = [".rst", ".md"]

templates_path = ['_templates']
exclude_patterns = []

language = 'es'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


import os
import sys
# Obtener la ruta absoluta al directorio raíz del proyecto
sys.path.insert(0, os.path.abspath('../..'))
# Ruta a los módulos del código
autodoc_mock_imports = []
