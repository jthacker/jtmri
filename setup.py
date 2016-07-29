from setuptools import find_packages, setup

exec(open('jtmri/_version.py').read())

setup(name='jtmri',
        version=__version__,
        packages=find_packages(),
        include_package_data = True,
        install_requires=[
            'arrview>=1.1',
            'arrow',
            'fuzzywuzzy',
            'h5py',
            'jinja2',
            'matplotlib',
            'numpy',
            'pandas',
            'prettytable',
            'PyYAML',
            'pydicom',
            'scikit-image',
            'scipy'],
        tests_require=[
            'pytest',
            ],
        setup_requires=['pytest-runner'])
