import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from distutils import log
from typing import List, Dict

# Constants
PROJECT_NAME = "enhanced_cs.SY_2508.10780v1_Learning_Task_Execution_Hierarchies_for_Redundant_"
VERSION = "1.0.0"
DESCRIPTION = "Enhanced AI project based on cs.SY_2508.10780v1_Learning-Task-Execution-Hierarchies-for-Redundant- with content analysis."
AUTHOR = "Alessandro Adami, Aris Synodinos, Matteo Iovino, Ruggero Carli, Pietro Falco"
EMAIL = "author@example.com"
URL = "https://example.com"
REQUIRES_PYTHON = ">=3.8.0"
REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "pandas",
]

# Setup configuration
class CustomInstallCommand(install):
    """Custom install command to handle additional setup tasks."""
    def run(self):
        install.run(self)
        log.info("Running custom install tasks...")

class CustomDevelopCommand(develop):
    """Custom develop command to handle additional setup tasks."""
    def run(self):
        develop.run(self)
        log.info("Running custom develop tasks...")

class CustomEggInfoCommand(egg_info):
    """Custom egg info command to handle additional setup tasks."""
    def run(self):
        egg_info.run(self)
        log.info("Running custom egg info tasks...")

def get_package_data() -> Dict[str, List[str]]:
    """Get package data for inclusion in the setup."""
    package_data = {
        "": ["*.txt", "*.md"],
    }
    return package_data

def get_package_dir() -> Dict[str, str]:
    """Get package directory for inclusion in the setup."""
    package_dir = {
        "": os.path.join(os.path.dirname(__file__), "src"),
    }
    return package_dir

def main():
    """Main setup function."""
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        python_requires=REQUIRES_PYTHON,
        packages=find_packages(where="src"),
        package_dir=get_package_dir(),
        package_data=get_package_data(),
        install_requires=REQUIRED_PACKAGES,
        include_package_data=True,
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand,
            "egg_info": CustomEggInfoCommand,
        },
    )

if __name__ == "__main__":
    main()