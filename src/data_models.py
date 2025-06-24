# filepath: src/data_models.py
"""
Data Models for Bayesian RCT Research
Data models for Bayesian RCT research analysis
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


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


@dataclass
class FisherInformationAnalysis:
    """Fisher information-based analysis results"""
    prior_type: str
    fisher_information_matrix: np.ndarray
    effective_sample_size: float
    information_gain_ratio: float
    theoretical_power_gain: float
    confidence_interval_reduction: float


@dataclass
class LLMConsistencyReport:
    """Comprehensive LLM consistency analysis"""
    parameter: str
    n_runs: int
    mean_estimates: List[float]
    std_estimates: List[float]
    overall_mean: float
    overall_std: float
    coefficient_of_variation: float
    confidence_interval_95: tuple
    stability_score: float  # 0-1, higher is more stable
    recommendation: str


@dataclass
class BayesianSampleSizeAnalysis:
    """Bayesian sample size analysis with theoretical grounding"""
    prior_type: str
    # Fisher information components
    prior_information_content: float
    expected_data_information: float
    posterior_information_content: float
    
    # Sample size calculations
    baseline_sample_size: int
    adjusted_sample_size: int
    sample_size_reduction_percent: float
    patients_saved: int
    
    # Power analysis
    baseline_power: float
    adjusted_power: float
    power_improvement: float
    
    # Theoretical justification
    theoretical_basis: str
    assumptions: List[str]
    confidence_level: float


@dataclass
class ComprehensiveAnalysisResults:
    """Complete analysis results structure"""
    # Basic components
    comparison_setup: Dict
    llm_priors: List[LLMPriorSpecification]
    
    # Enhanced analyses
    consistency_reports: List[LLMConsistencyReport]
    fisher_analyses: List[FisherInformationAnalysis]
    bayesian_sample_size_analyses: List[BayesianSampleSizeAnalysis]
    
    # Metadata
    analysis_timestamp: str
    n_llm_runs: int
    api_key_used: bool
    theoretical_validation: Dict
