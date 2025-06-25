# filepath: src/__init__.py
"""
Bayesian RCT Research Package
Bayesian RCT Research Package
"""

from .data_models import (
    LLMPriorSpecification,
    ConsistencyAnalysis,
    PriorComparison,
    SampleSizeComparison,
    BayesianSampleSizeAnalysis,
    LLMConsistencyReport,
    FisherInformationAnalysis
)

from .llm_elicitor import (
    ProductionLLMPriorElicitor,
    MockLLMPriorElicitor
)

from .data_loader import (
    load_historical_priors,
    load_actual_toenail_data
)

from .analysis import (
    comparative_analysis_setup,
    compare_prior_specifications,
    calculate_sample_size_benefits,
    calculate_bayesian_sample_size_analysis,
    analyze_llm_consistency
)

from .output_handler import (
    save_analysis_results,
    save_summary_report,
    save_prior_comparison_csv,
    create_output_directory_structure
)

from .posterior_predictive_evaluation import (
    PosteriorPredictiveEvaluator,
    PosteriorPredictiveResult,
    BenchmarkingResult,
    generate_synthetic_test_data
)

__all__ = [
    'LLMPriorSpecification',
    'ConsistencyAnalysis', 
    'PriorComparison',
    'SampleSizeComparison',
    'ProductionLLMPriorElicitor',
    'MockLLMPriorElicitor',
    'load_historical_priors',
    'load_actual_toenail_data',
    'comparative_analysis_setup',
    'compare_prior_specifications',
    'calculate_sample_size_benefits',
    'save_analysis_results',
    'save_summary_report',
    'save_prior_comparison_csv',
    'create_output_directory_structure',
    'PosteriorPredictiveEvaluator',
    'PosteriorPredictiveResult',
    'BenchmarkingResult',
    'generate_synthetic_test_data'
]
