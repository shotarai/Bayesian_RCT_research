# filepath: src/analysis.py
"""
Analysis Functions for Bayesian RCT Research
Analysis Functions for Bayesian RCT Research
"""

import numpy as np
import logging
from typing import Dict, List, Optional

from .data_models import PriorComparison, SampleSizeComparison
from .llm_elicitor import ProductionLLMPriorElicitor, MockLLMPriorElicitor
from .data_loader import load_historical_priors

logger = logging.getLogger(__name__)


def comparative_analysis_setup(api_key: Optional[str] = None):
    """
    Setup comparison between LLM priors and historical priors from previous studies
    Uses MockLLMPriorElicitor if no API key is provided
    """
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: LLM vs HISTORICAL PRIORS")
    print("="*80)
    
    # Obtain LLM priors (switch based on API key availability)
    if api_key:
        try:
            elicitor = ProductionLLMPriorElicitor(api_key=api_key)
        except Exception as e:
            logger.warning(f"⚠️ Production LLM failed: {e}")
            logger.info("🔄 Falling back to Mock LLM")
            elicitor = MockLLMPriorElicitor()
    else:
        logger.info("🤖 Using Mock LLM (no API key provided)")
        elicitor = MockLLMPriorElicitor()
    
    llm_priors = elicitor.elicit_clinical_priors(
        treatment_1="Itraconazole 250mg daily for 12 weeks",
        treatment_2="Terbinafine 250mg daily for 12 weeks",
        outcome_measure="unaffected nail length in millimeters", 
        clinical_context="toenail fungal infection (onychomycosis) in adults"
    )
    
    # Prior distributions from previous studies
    historical_priors = load_historical_priors()
    
    # Convert LLM priors
    llm_analysis_priors = elicitor.export_priors_for_analysis(llm_priors)
    
    comparison_setup = {
        'historical_fixed': historical_priors['fixed_effect_model'],
        'historical_mixed': historical_priors['mixed_effect_model'], 
        'llm_expert': llm_analysis_priors,
        'uninformative': {
            'beta_intercept': {'dist': 'normal', 'mu': 0, 'sigma': 100},
            'beta_time': {'dist': 'normal', 'mu': 0, 'sigma': 100}, 
            'beta_interaction': {'dist': 'normal', 'mu': 0, 'sigma': 100},
            'sigma': {'dist': 'half_normal', 'sigma': 100}
        }
    }
    
    return comparison_setup, llm_priors


def compare_prior_specifications(llm_priors: Dict, historical_priors: Dict) -> List[PriorComparison]:
    """
    Detailed comparison between LLM priors and historical priors
    """
    comparisons = []
    
    for param in ['beta_intercept', 'beta_time', 'beta_interaction']:
        if param in llm_priors and param in historical_priors:
            llm = llm_priors[param]
            hist = historical_priors[param]
            
            # Differences in mean and standard deviation
            diff_mean = llm['mu'] - hist['mu']
            diff_std = llm['sigma'] - hist['sigma']
            
            # Calculate overlap coefficient
            overlap = calculate_distribution_overlap(llm['mu'], llm['sigma'], hist['mu'], hist['sigma'])
            
            comparison = PriorComparison(
                parameter=param,
                llm_prior=llm,
                historical_prior=hist,
                difference_mean=diff_mean,
                difference_std=diff_std,
                overlap_coefficient=overlap
            )
            comparisons.append(comparison)
    
    return comparisons


def calculate_distribution_overlap(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """
    Calculate overlap coefficient between two normal distributions
    """
    combined_variance = (sigma1**2 + sigma2**2) / 2
    distance = abs(mu1 - mu2)
    overlap = np.exp(-0.25 * distance**2 / combined_variance)
    return overlap


def calculate_sample_size_benefits(comparison_setup: Dict) -> List[SampleSizeComparison]:
    """
    Compare information content of prior distributions and calculate sample size reduction effects
    """
    results = []
    
    # Calculate information content for each prior distribution
    for prior_name, priors in comparison_setup.items():
        if prior_name == 'uninformative':
            continue
            
        # Calculate prior precision (inverse of variance)
        beta_precision = sum([
            1/priors[param]['sigma']**2 
            for param in ['beta_intercept', 'beta_time', 'beta_interaction'] 
            if param in priors
        ])
        
        # Compare with uninformative priors
        uninformative_precision = sum([
            1/comparison_setup['uninformative'][param]['sigma']**2 
            for param in ['beta_intercept', 'beta_time', 'beta_interaction']
        ])
        
        # Calculate information gain
        information_gain = beta_precision / uninformative_precision if uninformative_precision > 0 else 1
        
        # Estimate sample size reduction effect
        sample_size_reduction = (information_gain - 1) / information_gain if information_gain > 1 else 0
        
        # Reduction from hypothetical baseline sample size
        baseline_n = 400  # Typical sample size for toenail fungal infection studies
        patients_saved = int(baseline_n * sample_size_reduction)
        
        results.append(SampleSizeComparison(
            prior_type=prior_name,
            effective_sample_size=information_gain,
            power=0.8 + 0.1 * information_gain,
            sample_size_reduction=sample_size_reduction,
            patient_savings=patients_saved
        ))
    
    return results
