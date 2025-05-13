from setuptools import setup
from pathlib import Path

# Read the requirements from requirements.txt
this_directory = Path(__file__).parent
requirements_path = this_directory / "requirements.txt"

with requirements_path.open() as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]


setup(
    name="ripples",
    version="1.0",
    description="Rippley rippling",
    packages=["ripples"],  # same as name
    install_requires=requirements,
)
