"""Setup file."""
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='action-mod-rllib',
    version='0.0.1',
    author='acxz',
    long_description=long_description,
    description='Override rllib policies with your own action modification',
    packages=setuptools.find_packages(),
    install_requires=[
        'ray[rllib]',
        'ray[tune]',
    ],
)
