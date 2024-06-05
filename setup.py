from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sneat',
    version='1.0.1',
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
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bhark/sNEAT',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)