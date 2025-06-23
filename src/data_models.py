# filepath: src/data_models.py
"""
Data Models for Bayesian RCT Research
ベイズRCT研究用のデータモデル
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class LLMPriorSpecification:
    """LLMから得られた事前分布の仕様"""
    parameter: str
    distribution: str
    mean: float
    std: float
    confidence: float
    rationale: str
    llm_model: str
    timestamp: str
    session_id: str


@dataclass
class ConsistencyAnalysis:
    """事前分布の一貫性解析結果"""
    parameter: str
    mean_values: List[float]
    std_values: List[float]
    mean_cv: float  # Coefficient of variation
    std_cv: float
    consistency_score: float
    recommendation: str


@dataclass
class PriorComparison:
    """事前分布の比較結果"""
    parameter: str
    llm_prior: Dict
    historical_prior: Dict
    difference_mean: float
    difference_std: float
    overlap_coefficient: float


@dataclass
class SampleSizeComparison:
    """サンプルサイズ比較結果"""
    prior_type: str
    effective_sample_size: float
    power: float
    sample_size_reduction: float
    patient_savings: int
