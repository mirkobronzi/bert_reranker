from setuptools import setup, find_packages


setup(
    name='bert_reranker',
    version='0.0.1',
    packages=find_packages(include=['bert_reranker', 'bert_reranker.*']),
    python_requires='>=3.7',
    install_requires=[
        'flake8', 'tqdm', 'pyyaml>=5.3', 'pytest', 'numpy',
        'torch', 'transformers', 'pytorch-lightning', 'sklearn'],
    entry_points={
        'console_scripts': [
            'main=bert_reranker.main:main'
        ],
    }
)
