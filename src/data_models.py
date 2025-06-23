# filepath: src/data_models.py
"""
Data Models for Bayesian RCT Research
Data models for Bayesian RCT research analysis
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class LLMPriorSpecification:
    """Prior distribution specifications obtained from LLM"""
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
    """Consistency analysis results for prior distributions"""
    parameter: str
    mean_values: List[float]
    std_values: List[float]
    mean_cv: float  # Coefficient of variation
    std_cv: float
    consistency_score: float
    recommendation: str


@dataclass
class PriorComparison:
    """Prior distribution comparison results"""
    parameter: str
    llm_prior: Dict
    historical_prior: Dict
    difference_mean: float
    difference_std: float
    overlap_coefficient: float


@dataclass
class SampleSizeComparison:
    """Sample size comparison results"""
    prior_type: str
    effective_sample_size: float
    power: float
    sample_size_reduction: float
    patient_savings: int
