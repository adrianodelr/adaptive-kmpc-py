from setuptools import setup, find_packages

setup(
    name="adaptive-kmpc-py",
    version="0.1.0",
    author="Adriano del Rio",
    author_email="adriano.delrio@gmx.net",
    description="Python implementation of the adaptive Koopman model predictive control algorithm",
    long_description=open("README.md").read(),
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)