import pathlib
from setuptools import setup
import os
import sys
import re
from pathlib import Path

# The directory containing this file
current_path = pathlib.Path(__file__).parent

# The text of the README file
readme_path = (current_path / "README.md").read_text()

install_requires = ['dtaidistance', 'matplotlib',
                    'pygmt', 'pandas', 'numpy', 'xarray']
setup_requires = ['setuptools>=18.0', 'cython>=0.29.6']

current_path = Path(__file__).parent
# Check version number
init_fn = current_path / 'dtwhaclustering' / '__init__.py'
with init_fn.open('r', encoding='utf-8') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)
if not version:
    raise RuntimeError('Cannot find version information')

readme_path = current_path / 'README.md'
if os.path.exists(readme_path):
    with readme_path.open('r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = ""

# This call to setup() does all the work
setup(
    name="dtwhaclustering",
    version=version,
    description="Codes to perform Dynamic Time Warping Based Hierarchical Agglomerative Clustering of GPS data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/earthinversion/DTW-based-Hierarchical-Clustering",
    author="Utpal Kumar",
    author_email="utpalkumar50@gmail.com",
    python_requires='>=3.5',
    license='Apache 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    packages=["dtwhaclustering"],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    keywords='dynamic time warping clustering',
    extras_require={
        'vis': ['matplotlib', 'pygmt'],
        'numpy': ['numpy', 'scipy'],
        'all': ['matplotlib', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'xarray']
    },
)
