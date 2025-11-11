from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\\n')
    return requirements

setup(
    name='JengaAI',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    author='JengaAI',
    author_email='contact@jengaai.com',
    description="JengaAI: Multi-Task NLP Framework for Sustainable Development & Security",
    long_description=open('README.MD').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Rogendo/JengaAI',  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
