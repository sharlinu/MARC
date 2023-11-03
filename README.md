# Summary
This is an implementation of a R-GCN based MAAC architecture in Pytorch.


# Requirements used 
- Python 3.9
- Pytorch 1.13.1
- Pytorch Geometric 2.2.0
- Numpy 1.24.0
- Baselines 0.1.5
- my [fork](https://github.com/sharlinu/lb-foraging) of the Level-based Foraging environment


# Experiment Settings
The algorithm and environment can be configured in the `config.yaml`. 

## Algorithm Configuration
This file contains all common parameters we would like to set, 
e.g. n_episodes, episode_lenght, batch size etc. Beyond the standard parameters, we have the following additional ones:
```yaml
dense : True (default) # determines if entities to be considered are only the objects and agents (i.e. True) or all cells in the grid (dense=False)
relational_embedding : False #Message Passing Network with relational embeddings or standard R-GCN
```

## Environment Configuration
As for the environment parameters, some are used
for all environments like `player: X` setting the number of agents to $X$ and `field: n` spawns a $n \times n$ grid. 

For **level-based foraging environment**, we have additional parameters of
```yaml
max_food: 3 # determines the number of apple to spawn 
force_coop : True # if set to True, the agents need to cooperate in order to pick up food 
max_player_level : 2 (default) # maximum level that players have
keep_food: False # whether or not the food stays on the grid after it is picked up. If it stays, it gets a level of -1, otherwise it is removed
grid_observation: True # the format of the observation as grid or 1D array. Needs to be True for MARC
```


# Usage
To launch experiments (training + evaluation) over a number of seeds just set config.yaml and run:
```commandline
$ python3 main_lbf.py
```

# Evaluation
After a model has run, it can be found in the folder `experiments/{alg}/{env_name}/{experiment_id}/saved_models/model_name.pth.tar`. 
For example, to evaluate MAAC we would run:
```commandline
$ python3 evaluate_maac.py experiments/MAAC/lbf/2023-09-27_lbf_15x15_8p_1f_coop_std_seed4001/saved_models/ckpt_final.pth.tar
```
