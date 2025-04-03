import os
from setuptools import setup, find_packages

# Add description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Parse requirements from requirements.txt
with open(os.path.join(current_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='llm_trainer',
    packages=find_packages('.'),
    version='0.1.22',
    license='MIT',
    description='🤖 Train your LLMs with ease and fun .',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Nikoláy Skripkó',
    author_email='nskripko@icloud.com',
    url='https://github.com/Skripkon/llm_trainer',
    keywords=[],
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.11',
)
