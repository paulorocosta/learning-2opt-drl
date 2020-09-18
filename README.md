# Learning 2-opt Heuristics for the TSP via Deep Reinforcement Learning


Implementation of the Policy Gradient algorithm for learning 2-opt improvement heuristics, following https://arxiv.org/abs/2004.01608

Dependencies: 
- Python 3.6.4
- Torch
- Numpy
- Matplotlib
- Apex
- tqdm
- pyconcorde

## How to test it?

To use the learned polcies reported in the paper you can run:
```
python TestLearnedAgent.py --load_path best_policy/policy-TSP20-epoch-189.pt --n_points 20 --test_size 1 --render 
```
where ``` load_path ``` can be replaced with one of the policies in /best_policy. 

## Results

Learned policy on a TSP with 50 nodes:

![Alt Text](data/tsp50.gif)
