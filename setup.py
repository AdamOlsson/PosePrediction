from setuptools import setup, find_packages

install_requires = [
    "torch",
    "torchvision",
    "pandas",
    "opencv-python",
    "numpy",
    "matplotlib"
]

packages = [
    "Transformers",
    "paf",
    "paf.pafprocess",
    "Datasets",
    "model",
    "model.FeatureExtractors",
    "util"
]

setup(name='PosePrediction',
      version='0.1',
      description='Human pose prediction using graphs',
      author='Adam Olsson',
      #author_email='',
      #url='https://www.python.org/sigs/distutils-sig/',
      install_requires=install_requires,
      packages=find_packages(),
     )