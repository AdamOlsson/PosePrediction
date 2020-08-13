
# Overview
A python package for performing pose predictions using [PyTorch](https://pytorch.org/). The code in this package is based on the research in [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050) which eventually became [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), a more extensive package for pose predictions. This package serves as a lightweight version of OpenPose with the bare minimum features. The motivation is that this package provides a fast and easy installation for anyone who whishes to get going with pose predictions.
<center><img src="docs/snatch.gif"></center>

# Installation

Requires swig which can easily be installed on Linux using:

```
sudo apt-get install swig
```

Once swig is installed and the repository is cloned, run:

```
python3 setup.py install
```

from the root directory. The setup script will install python dependencies and build binaries for the Part Affinity Fields code. If the user would like to use pre-trained network weights, [git-lfs](https://github.com/git-lfs/git-lfs/wiki/Installation) needs to be installed. To install git-lfs on Linux run:

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
```

```
sudo apt-get install git-lfs
```

```
git lfs install
```
From this repository, pull the network weights using:
```
git lfs fetch && git lfs pull
```

**Note**: On some systems git-lfs can force user to enter their username and password for every file they wish to push. Read more about this issue [here](https://github.com/git-lfs/git-lfs/issues/2014). Should this happen, run:

```
git config --global credential.helper 'cache --timeout=3600'
```

# Limitations

As of now, there are some limitations to what this package can do. The driving force to solve this limitations is my need of having these features.

* **Training** - As of now, it is not possible to perform any typ of training the model. The current weights are the results from the paper [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050) and returns satisfying predictions.

* **Inefficient Drawing of Poses** - The current method for drawing poses in videos is inefficient because it can only draw them frame by frame. Instead, passing the entire video as a batch and perform batch operations would significantly reduce the processing time. However, since this package is not developed to work in real-time situations, this is fine.

# Big Thanks To

A lot of inspiration when developing this package has been taken from
[github1](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation) and [github2](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation). Additionally, the code for the Part Affinity fields computations has been copied from [this](https://github.com/ildoonet/tf-pose-estimation) repository and can be found [here](https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess).