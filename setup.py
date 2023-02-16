from setuptools import setup, find_packages

setup(
    name='sentiment_analysis_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'nltk',
        'scikit-learn',
        'bs4',
        'requests',
        
    ],
    description='A simple package for sentiment analysis'
)
