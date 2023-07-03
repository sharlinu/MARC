# Summary
This is an implementation of a R-GCN based MAAC architecture in Pytorch.


# Requirements used 
- Python 3.9
- Pytorch 1.13.1
- tensorflow 2.10.0 (only needed for tensorboard implementation so older versions possible)
- TensorboardX
- Numpy 1.24.0
- Baselines
- 

# Usage
To launch experiments (training + evaluation) over a number of seeds just set config.yaml and run:
```
$ python rel_main.py
$ python main_lbf.py
```
