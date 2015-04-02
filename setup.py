from distutils.core import setup
from setuptools import find_packages

setup(
        name='jtmri',
        version='0.1dev',
        packages=find_packages(),
        include_package_data = True,
        install_requires=[
            'numpy','traits','traitsui',
            'scikit-image','tqdm',
            'prettytable', 'PySide', 'PyYAML',
            'pydicom', 'fuzzywuzzy', 'jinja2']
)
