from setuptools import setup

setup(
    name='hypercube-sampler',
    version='0.0.1',
    py_modules=['hypercube-sampler'],
    install_requires=[
        'click',
        'numpy',
        'scipy'
    ],
    entry_points={
        'console_scripts': ["sampler=hypercube_sampler.cli:main"]
    },
)
