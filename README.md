# P1 Navigation
 This repository contains my implementation of the first project of Udacity's Reinforcement Learning Nanodegree: Navigation.
 It consists in the application of a Double Deep Q-Learning algorithm to an environment of bananas.

# Installation
Clone this GitHub repository, create a new conda environment and install all the required libraries using the frozen `conda_requirements.txt` file.
```shell
$ git clone https://github.com/dimartinot/P1-Navigation.git
$ cd P1-Navigation/
$ conda env create --file conda_requirements.txt
``` 

If you would rather use pip rather than conda to install libraries, execute the following:
```shell
$ pip install -r requirements.txt
```
*In this case, you may need to create your own environment (it is, at least, highly recommended).*

# Usage
Provided that you used the conda installation, your environment name should be *drlnd*. If not, use `conda activate <your_env_name>` 
```shell
$ conda activate drlnd
$ jupyter notebook
```

# Code architecture
This project is made of two main jupyter notebooks using classes spread in multiple Python file:
## Jupyter notebooks
- `Navigation.ipynb`: notebook of the code execution. It loads the weight file of the model and execute 5 runs of the environment while averaging, at the end, the score obtained in all these environment (make sure to have provided the correct path to the Banana exe file). 
- `project_nagivation.ipynb`: notebook containing the training procedure (see especially the `dqn` method for details).

## Python files
 - `agent.py`: Contains the class definition of the basic double q learning algorithm that uses soft update (for wieght transfer between the local and target networks) as well as a uniformly distributed replay buffer
 - `model.py`: Contains the PyTorch class definition of a neural network, used by the target and local networks.
 - `prioritizedAgent.py`: Class inheriting from the *basic* `Agent` class, adding a priority measure for sampling experience tuples from the replay buffer. The probability for an experience tuple `(s_i, a_i, r_i, s_i')` is given by: 
 <div style="text-align:center">
    <img width="100px" src="https://render.githubusercontent.com/render/math?math=\frac{p^{\alpha}}{\sum_{i} p_{i}^{\alpha}}"> 
 </div>
where p is the lastly measured td error of the experience tuple, before its insertion in the buffer. Alpha is an hyperparameter: the closer alpha is to zero, the more uniform the sampling distribution will be. The closer to one, the less uniform it will be. As a default, the alpha value used in this project was 0.9.

## PyTorch weights
Two weight files ar provided, one for each agent: as they implement the same model, they are interchangeable. However, as a different training process has been used, I found it interesting to compare the resulting behaviour of the agent:
 - `checkpoint_double_q_learning.pth`: these are the weights of a *common* double q learning agent using a uniformly distributed replay buffer.
 - `checkpoint_prioritized_double_q_learning.pth`: these are the weights of an agent trained with experience tuple sampled using their *importance* in the learning process.
