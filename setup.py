from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
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
    install_requires=get_requirements('requirements.txt')
    
)
