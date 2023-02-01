# Summary
This is an alternative maac implementation in pytorch that also takes in episodic tasks and does not rely on old package versions anymore

# Requirements used 
- Python 3.9
- [multi-agent-particle-env](https://bitbucket.org/epesce/multi-agent-particle-env/src/master/) (my fork)
- Pytorch 1.13.1
- tensorflow 2.10.0 (only needed for tensorboard implementation so older versions possible)
- TensorboardX
- Numpy 1.24.0
- Baselines

# Usage
To launch experiments (training + evaluation) over a number of seeds just set scenario and seeds in 'main_seeds.py' and run it:
```
$ python main_seeds.py
```
Experiments settings can be changed editing the parameters file: ``params_seeds.py``