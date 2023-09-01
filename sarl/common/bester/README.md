The following is an excerpt from the README.md of this folder's primary source: https://github.com/cycraig/MP-DQN

#  Multi-Pass Deep Q-Networks

This folder includes several reinforcement learning algorithms for parameterised action space MDPs:

1. P-DQN [[Xiong et al. 2018]](https://arxiv.org/abs/1810.06394)

    - MP-DQN [[Bester et al. 2019]](https://arxiv.org/abs/1905.04388)
    - SP-DQN [[Bester et al. 2019]](https://arxiv.org/abs/1905.04388)
   
2. PA-DDPG [[Hausknecht & Stone 2016]](https://arxiv.org/abs/1511.04143)
3. Q-PAMDP [[Masson et al. 2016]](https://arxiv.org/abs/1509.01644)

Multi-Pass Deep Q-Networks (MP-DQN) fixes the over-paramaterisation problem of P-DQN by splitting the action-parameter inputs to the Q-network using several passes (in a parallel batch). Split Deep Q-Networks (SP-DQN) is a much slower solution which uses multiple Q-networks with/without shared feature-extraction layers. A weighted-indexed action-parameter loss function is also provided for P-DQN.


## Citing
If this collection has helped your research, please cite the following:

```bibtex
@article{bester2019mpdqn,
	author    = {Bester, Craig J. and James, Steven D. and Konidaris, George D.},
	title     = {Multi-Pass {Q}-Networks for Deep Reinforcement Learning with Parameterised Action Spaces},
	journal   = {arXiv preprint arXiv:1905.04388},
	year      = {2019},
	archivePrefix = {arXiv},
	eprinttype    = {arxiv},
	eprint    = {1905.04388},
	url       = {http://arxiv.org/abs/1905.04388},
}
```
