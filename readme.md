# Optimization Based Robotics

Final project: implementing Deep Q-Learning.

# Acknowledgment

This is a project for the course **Optimization Based Robotics** of the University of Trento.

Professors:
- Andrea Del Prete
- Gianluigi Grandesso

Students:
- Giovanni Lorenzini
- Simone Luchetta

# Overview

The aim of this project is to train a robot composed of `n` joints via Deep Q-Learning (DQN). The task is considered fullfilled if the robot can reach a balancing upward pose.

Amongst the families of DQN approaches, we decided to implement the Network by passing only the states of the robot `x`, hence joint angle configurations `q` and joint velocities `v`. The output of the Network is going to be a list of approximate Q-values, one for each possible action `u`.

The original pendulum class is overridden with a custom one to suit the Network's needs.
In particular, states are kept continuous and are not discretized, whereas the actions `u` are converted from discrete (best Q-value index corresponding to the output of the net) to continuous.

The Network's structure is kept fairly simple and is reported here below:

aaaaaaaaaaaaaa

A capture of the result is also available hereinafter:

aaaaaaaaaaaaaa

# Preparation

## Simulator

Install tools for programming in python:

```shell
$ sudo apt-get update
$ sudo apt install terminator python3-numpy python3-scipy python3-matplotlib spyder3 curl
```

Add to the apt source list the source for the simulator:

```shell
$ sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
$ sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
$ curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add 
$ sudo apt-get update
```

Install pinocchio and the simulator:

```shell
$ sudo apt install robotpkg-py38-pinocchio robotpkg-py38-example-robot-data 
$ robotpkg-urdfdom robotpkg-py38-qt5-gepetto-viewer-corba robotpkg-py38-quadprog robotpkg-py38-tsid
```

Add the following lines to your `~/.bashrc` file:

```shell
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.6/site-packages
export PYTHONPATH=$PYTHONPATH:<folder_containing_orc>
```

## PyTorch

The framework used for this project is `PyTorch`. It has been installed with the following command:

```shell
$ pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

This is going to install the stable version for Linux using the pip packet manager.
The code is in `Python`, and the Network will be trained using the CPU.

## Repository clone

Clone the repository inside the `orc` folder:

```shell
$ cd ~/orc
$ git clone https://github.com/lorenzinigiovanni/orc-project.git
```

# Execution

```shell
$ cd ~/orc/orc-project/
$ python3 main.py
```

# Resources

- Overview of DQL in Pytorch: https://towardsdatascience.com/deep-q-network-with-pytorch-146bfa939dfe
- Pytorch resources: https://pytorch.org
- DQL background: https://www.tensorflow.org/agents/tutorials/0_intro_rl
