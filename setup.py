from setuptools import setup, find_packages

setup(
    name='stacked_capsule_autoencoders',
    version=1.0,
    packages=find_packages(),
    install_requires=[
        'absl_py>=0.8.1',
        'imageio>=2.6.1',
        'matplotlib>=3.0.3',
        'monty>=3.0.2',
        'numpy>=1.16.2',
        'Pillow>=6.2.1',
        'scikit_learn>=0.20.4',
        'scipy>=1.2.1',
        'dm_sonnet==1.35',
        'tensorflow==2.5.3',
        'tensorflow_probability==0.8.0',
        'tensorflow_datasets==1.3.0',
    ],
)
