from setuptools import setup, find_packages

setup(
    name="adaptive-kmpc-py",
    version="0.1.0",
    author="Adriano del Rio",
    description="Python implementation of the adaptive Koopman model predictive control algorithm",
    long_description=open("README.md").read(),
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"adaptive_kmpc_py": ["py.typed"]}
)