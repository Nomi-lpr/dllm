"""
C-Eval 评估指标

与 MMLU 同为四选一（A/B/C/D），答案格式与判断逻辑一致，直接复用 MMLUMetrics。
"""

from .mmlu_metrics import MMLUMetrics


class CevalMetrics(MMLUMetrics):
    """
    C-Eval 任务的评估指标（逻辑与 MMLU 一致：从生成文本提取 A/B/C/D，与 ground_truth 比较）
    """

    pass
