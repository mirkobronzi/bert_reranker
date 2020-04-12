from setuptools import setup, find_packages


setup(
    name='bert_reranker',
    version='0.0.1',
    packages=find_packages(include=['bert_reranker', 'bert_reranker.*']),
    python_requires='>=3.7',
    install_requires=[
        'flake8', 'tqdm', 'pyyaml>=5.3', 'pytest', 'numpy>=1.16.4', 'pytest',
        'torch', 'transformers', 'pytorch-lightning', 'sklearn', 'pandas', 'setuptools>=41.0.0'],
    entry_points={
        'console_scripts': [
            'main=bert_reranker.main:main'
        ],
    }
)
