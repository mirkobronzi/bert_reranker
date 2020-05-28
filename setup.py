from setuptools import setup, find_packages


setup(
    name='bert_reranker',
    version='0.2.0',
    packages=find_packages(include=['bert_reranker', 'bert_reranker.*']),
    license='MIT',
    author='Mirko Bronzi',
    author_email='m.bronzi@gmail.com',
    url='https://github.com/mirkobronzi/bert_reranker',
    python_requires='>=3.7',
    install_requires=[
        'flake8', 'tqdm', 'pyyaml>=5.3', 'pytest', 'numpy>=1.16.4', 'pytest', 'pandas', 'nltk',
        'torch==1.4.0', 'transformers==2.8.0', 'pytorch-lightning==0.7.6rc1', 'scikit-learn',
        'pandas', 'wandb', 'setuptools>=41.0.0',
        'orion @ git+git://github.com/Epistimio/orion.git@develop#egg=orion'],
    entry_points={
        'console_scripts': [
            'main=bert_reranker.main:main',
            'train_outlier_detector=bert_reranker.models.sklearn_outliers_model:main'
        ],
    }
)
