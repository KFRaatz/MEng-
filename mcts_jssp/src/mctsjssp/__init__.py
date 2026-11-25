"""MCTS with fixed and learned rollout distributions for the classical job shop scheduling problem.

The package follows Chapter 3 of the thesis:

==================  =====================================================================
Sub-package         Thesis section
==================  =====================================================================
``instances``       3.5  JSSP data (OR-Library test set, generated instances, disturbances)
``env``             3.3  dynamics model, 3.4 environment, 3.7 schedule checker
``distributions``   3.2.2 rollout distributions, entropy levels and their temperatures
``mcts``            3.2.1 search procedure (UCT), PUCT for self-play, recording (3.6.4)
``networks``        3.6.2 GIN of LR and fully connected networks of AZ
``training``        3.6.1 self-play training, 3.6.2 tuning and offline assessment
``experiments``     3.6.3 experimental design (RQ1-RQ5), pilot study, 3.7 validation
``analysis``        3.6.4 performance measures and statistical analysis
==================  =====================================================================
"""

__version__ = "1.0.0"
