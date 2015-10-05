from distutils.core import setup
from setuptools import find_packages
from jtmri import __version__

setup(
        name='jtmri',
        version=__version__,
        packages=find_packages(),
        include_package_data = True,
        install_requires=[
            'numpy','traits','traitsui',
            'scikit-image','tqdm',
            'prettytable', 'PySide', 'PyYAML',
            'pydicom', 'fuzzywuzzy', 'jinja2',
            'arrow']
)
