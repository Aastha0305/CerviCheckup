from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    this function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="cervical-cancer-prediction",
    version="0.1.0",
    author="Aastha Luthra",
    author_email="aasthaluthraa@gmail.com",
    description="I tested several algorithms to fine tune a model that can predict whether a women has cervical cancer or not",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aastha0305/CerviCheckup",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
