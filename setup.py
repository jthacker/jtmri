from setuptools import find_packages, setup

exec(open('jtmri/_version.py').read())

setup(name='jtmri',
      version=__version__,
      packages=find_packages(),
      include_package_data = True,
      install_requires=[
        'numpy',
        'scikit-image',
        'prettytable',
        'PyYAML',
        'matplotlib',
        'pydicom',
        'fuzzywuzzy',
        'jinja2',
        'arrow',
        'pandas'])
