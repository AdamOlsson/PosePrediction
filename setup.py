from distutils.core import setup

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
      version='1.0',
      description='Human pose prediction using graphs',
      author='Adam Olsson',
      #author_email='',
      #url='https://www.python.org/sigs/distutils-sig/',
      install_requires=install_requires,
      packages=packages,
     )