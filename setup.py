from setuptools import setup, find_packages

setup(
    name="master",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True
)