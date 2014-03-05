from distutils.core import setup

setup(
        name='jtmri',
        version='0.1dev',
        packages=['jtmri', 'jtmri.dcm',
            'jtmri.scripts'],
        install_requires=[
            'numpy','traits','traitsui',
            'scikit-image','tqdm',
            'prettytable', 'PySide', 'PyYAML']
)
