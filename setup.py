from setuptools import setup, find_packages
from typing import List

#HYPEN_E_DOT- '-e .'
def get_requirements(file_path='requirements.txt')-> List[str]:
    """    Returns A list of requirement strings.
    """
    requirements = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip any leading/trailing whitespace and ignore empty lines or comments
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        print(f"Requirements file {file_path} not found.")
    return requirements


setup(
    name='cervical-cancer-prediction',                      # Replace with your project name
    version='0.1.0',                           # Replace with your version
    author='Aastha Luthra',                       # Replace with your name
    author_email='aasthaluthraa@gmail.com',    # Replace with your email
    description='We tested several models to predict whether a women has cervical cancer or not',
    #long_description=open('README.md').read(),  # Read the content of README.md
    #long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/my_ml_project',  # Replace with your project's URL
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=get_requirements('Requirements.txt')
    
)
