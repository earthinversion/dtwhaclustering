#!/bin/bash
rm -r build dtwhaclustering.egg-info dist
python setup.py sdist bdist_wheel
twine check dist/*
twine upload --repository-url https://test.pypi.org/legacy/ dist/*