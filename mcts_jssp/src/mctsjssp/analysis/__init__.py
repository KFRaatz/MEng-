"""Performance measures and statistical analysis (Section 3.6.4)."""
from .metrics import bks_hits, convergence, instance_means, load_frame, summarise, time_to_target, value_to_reach
from .report import write_report
from .rq import analyse
from .stats import (compare_many, compare_pairs, friedman_test, holm, rank_posthoc, signed_rank_sums,
                    wilcoxon_signed_rank)

__all__ = ["bks_hits", "convergence", "instance_means", "load_frame", "summarise", "time_to_target",
           "value_to_reach", "write_report", "analyse", "compare_many", "compare_pairs", "friedman_test", "holm",
           "rank_posthoc", "signed_rank_sums", "wilcoxon_signed_rank"]
