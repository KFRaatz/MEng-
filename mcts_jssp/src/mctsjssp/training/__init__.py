"""Self-play training, tuning and offline assessment of the networks (Sections 3.6.1 and 3.6.2)."""
from .imitation import (imitate, imitate_fast, network_feature_records, rule_episodes,
                        rule_feature_records)
from .ranked_reward import RankedReward
from .replay import Record, ReplayBuffer
from .selfplay import EpisodeRunner, play_puct_episode, play_uct_episode
from .trainer import TrainConfig, Trainer, greedy_ratio, uniform_ratio

__all__ = ["imitate", "imitate_fast", "network_feature_records", "rule_feature_records", "rule_episodes", "RankedReward", "Record", "ReplayBuffer", "EpisodeRunner", "play_puct_episode", "play_uct_episode",
           "TrainConfig", "Trainer", "greedy_ratio", "uniform_ratio"]
