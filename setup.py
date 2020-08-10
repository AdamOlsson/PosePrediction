from setuptools import setup, find_packages
import os

install_requires = [
    "torch",
    "torchvision",
    "pandas",
    "opencv-python",
    "numpy",
    "matplotlib",
    "scipy",
    "av"
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

# Building pafprocess
print("Building paf lib...")
stream = os.popen('bash paf/pafprocess/make.sh')
output = stream.read()
print(output)