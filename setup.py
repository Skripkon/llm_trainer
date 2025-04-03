import os
from setuptools import setup, find_packages

# Add description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    pass

# Parse requirements from requirements.txt
try:
    with open(os.path.join(current_directory, 'requirements.txt'), encoding='utf-8') as f:
        requirements = f.read().splitlines()
except Exception:
    print("Exception occurred during parsing requirements.txt")
    requirements = [
        'torch==2.6.0',
        'numpy==2.2.3',
        'tiktoken==0.9.0',
        'datasets==3.3.2',
        'tqdm==4.67.1',
    ]

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
