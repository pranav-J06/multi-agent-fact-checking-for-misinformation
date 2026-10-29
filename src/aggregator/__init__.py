"""Aggregator package."""
from src.aggregator.feature_extractor import FeatureExtractor
from src.aggregator.model import VerdictAggregatorMLP
from src.aggregator.trainer import train_aggregator

__all__ = ["FeatureExtractor", "VerdictAggregatorMLP", "train_aggregator"]
