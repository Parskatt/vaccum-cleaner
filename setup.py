from setuptools import setup

setup(
    name="vacuum_cleaner",
    version="0.0.1",
    author="Johan Edstedt",
    install_requires=open("requirements.txt", "r").read().split("\n"),
    packages=["vacuum_cleaner"],
)
