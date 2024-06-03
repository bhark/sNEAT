from setuptools import setup, find_packages

setup(
    name='sneat',
    version='0.0.5',
    packages=find_packages(),
    package_data={'sneat': ['default_config.ini']},
    install_requires=[
        'matplotlib==3.9.0',
        'networkx==3.3',
        'numpy==1.26.4',
        'tabulate==0.9.0',
        'tqdm==4.66.4'
    ],
    extras_require={
        'examples': [
            'gymnasium',
            'gymnasium[box2d]',
            'gymnasium[classic_control]',
        ]
    },
    author='Adrian E. Bratlann',
    author_email='aeb@tetrabit.coop',
    description='Simplified implementation of NEAT (Neuro-Evolution of Augmenting Topologies).',
    license='GPLv3'
)