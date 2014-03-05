from distutils.core import setup

setup(
        name='jtmri',
        version='0.1dev',
        packages=['jtmri'],
        install_requires=[
            'numpy','traits','traitsui',
            'scikit-image','tqdm',
            'prettytable', 'PySide', 'PyYAML']
)
