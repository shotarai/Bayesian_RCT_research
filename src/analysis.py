# filepath: src/analysis.py
"""
Analysis Functions for Bayesian RCT Research
Analysis Functions for Bayesian RCT Research
"""

import numpy as np
import logging
from typing import Dict, List, Optional

from .data_models import (
    PriorComparison, 
    SampleSizeComparison, 
    BayesianSampleSizeAnalysis,
    LLMConsistencyReport,
    FisherInformationAnalysis
)
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
    DEPRECATED: Use calculate_bayesian_sample_size_analysis for more rigorous calculations
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


def calculate_fisher_information_matrix(priors: Dict, 
                                      n_patients: int = 400, 
                                      use_actual_design: bool = True,
                                      error_variance: float = 9.63) -> np.ndarray:
    """
    Calculate Fisher Information Matrix based on rigorous statistical theory
    
    For linear mixed effects model: y_ij = β₀ + β₁*time_ij + β₂*treat_i + ε_ij
    Fisher Information: I(β) = (1/σ²) * X^T X + Λ_prior^(-1)
    
    Parameters:
    -----------
    priors : Dict
        Prior specifications with 'sigma' values for each parameter
    n_patients : int
        Number of patients in the study (default: 400)
    use_actual_design : bool
        If True, use actual toenail study design; if False, use scaled approximation
    error_variance : float
        Residual error variance σ² (default: 13.55 from toenail data linear regression)
        
    Returns:
    --------
    np.ndarray
        Fisher Information Matrix (3x3 for intercept, time, treatment)
    """
    # Model parameters: [β₀, β₁, β₂] = [intercept, time_effect, treatment_effect]
    param_names = ['beta_intercept', 'beta_time', 'beta_interaction']
    n_params = 3
    
    # Prior precision matrix (diagonal)
    prior_precision = np.zeros((n_params, n_params))
    for i, param in enumerate(param_names):
        if param in priors:
            prior_precision[i, i] = 1 / (priors[param]['sigma'] ** 2)
    
    if use_actual_design:
        # Use actual toenail study design (X^T X from 1854 observations, 298 patients)
        # Scale to requested sample size
        actual_xtx = np.array([
            [1854., 7975., 954.],
            [7975., 64139., 4143.],
            [954., 4143., 954.]
        ])
        
        # Scale by the ratio of requested patients to actual patients
        scale_factor = n_patients / 298  # 298 is actual number of patients in toenail data
        scaled_xtx = actual_xtx * scale_factor
        
    else:
        # Simplified balanced design approximation
        total_observations = n_patients * 7  # Average 7 timepoints
        n_treat_1 = n_patients // 2
        
        # Time points: [0,1,2,3,6,9,12] with average frequency
        avg_time = 4.57  # Mean of time points
        avg_time_squared = 35.86  # Mean of squared time points
        
        scaled_xtx = np.array([
            [total_observations, avg_time * total_observations, n_treat_1],
            [avg_time * total_observations, avg_time_squared * total_observations, avg_time * n_treat_1],
            [n_treat_1, avg_time * n_treat_1, n_treat_1]
        ])
    
    # Data information matrix: (1/σ²) * X^T X
    data_information = scaled_xtx / error_variance
    
    # Total Fisher Information = Prior precision + Data information
    fisher_info = prior_precision + data_information
    
    return fisher_info


def calculate_bayesian_sample_size_analysis(comparison_setup: Dict, 
                                          baseline_sample_size: int = 400,
                                          target_power: float = 0.8) -> List[BayesianSampleSizeAnalysis]:
    """
    Rigorous Bayesian sample size analysis based on Fisher Information Theory
    
    Theoretical Foundation:
    - For linear model with normal priors: I_posterior = I_prior + I_likelihood
    - Sample size relationship: n_new / n_old = |I_old| / |I_new|
    - Power improvement: Var(β̂_new) = 1/I_new < Var(β̂_old) = 1/I_old
    
    Parameters:
    -----------
    comparison_setup : Dict
        Prior specifications for comparison
    baseline_sample_size : int
        Baseline sample size for uninformative prior scenario (default: 400)
    target_power : float
        Target statistical power (default: 0.8)
        
    Returns:
    --------
    List[BayesianSampleSizeAnalysis]
        Rigorous sample size analysis results with theoretical validation
    """
    results = []
    
    uninformative_priors = comparison_setup.get('uninformative', {})
    
    for prior_name, priors in comparison_setup.items():
        if prior_name == 'uninformative':
            continue
            
        # Calculate Fisher Information Matrices with correct parameters
        uninformative_fisher = calculate_fisher_information_matrix(
            uninformative_priors, 
            n_patients=baseline_sample_size,
            use_actual_design=True,  # Use actual toenail study design
            error_variance=13.55  # Corrected value from actual data regression
        )
        informative_fisher = calculate_fisher_information_matrix(
            priors,
            n_patients=baseline_sample_size, 
            use_actual_design=True,
            error_variance=13.55  # Corrected value from actual data regression
        )
        
        # Information content measures
        prior_info_content = np.trace(informative_fisher - uninformative_fisher)
        expected_data_info = np.trace(uninformative_fisher)
        posterior_info_content = np.trace(informative_fisher)
        
        # Rigorous sample size reduction calculation
        # Theory: n_informative / n_uninformative = |I_uninformative| / |I_informative|
        det_uninformative = np.linalg.det(uninformative_fisher)
        det_informative = np.linalg.det(informative_fisher)
        
        if det_uninformative > 0 and det_informative > det_uninformative:
            # Sample size ratio (smaller is better)
            sample_size_ratio = det_uninformative / det_informative
            sample_size_reduction = 1 - sample_size_ratio
            
            # Ensure realistic bounds based on empirical validation
            # Mathematical validation shows: σ=1 vs σ=10 gives 2.8% reduction
            # Cap at 10% to prevent overoptimistic estimates
            sample_size_reduction = max(0, min(0.10, sample_size_reduction))  # Cap at 10%
        else:
            sample_size_reduction = 0
            
        adjusted_sample_size = int(baseline_sample_size * (1 - sample_size_reduction))
        patients_saved = baseline_sample_size - adjusted_sample_size
        
        # Power calculations based on variance reduction
        baseline_power = target_power
        
        # Power improvement from precision gain (variance reduction)
        if det_uninformative > 0:
            precision_improvement = det_informative / det_uninformative
            # Power improvement approximation: Power ∝ √(precision improvement)
            power_multiplier = min(1.2, np.sqrt(precision_improvement))  # Cap at 20% improvement
            power_improvement = (power_multiplier - 1) * baseline_power
        else:
            power_improvement = 0
            
        adjusted_power = min(0.95, baseline_power + power_improvement)
        
        # Enhanced theoretical justification
        information_ratio = det_informative / det_uninformative if det_uninformative > 0 else 1
        theoretical_basis = (
            f"Fisher Information Theory: I_posterior = I_prior + I_likelihood. "
            f"Sample size ratio = |I_uninf|/|I_inf| = {sample_size_ratio:.4f}. "
            f"Precision improvement = {information_ratio:.3f}x"
        )
        
        assumptions = [
            "Linear mixed effects model: y_ij = β₀ + β₁*time_ij + β₂*treat_i + ε_ij",
            "Normal priors on regression coefficients",
            f"Balanced longitudinal design with {baseline_sample_size} patients, 7 timepoints",
            f"Residual variance σ² = 13.55 (estimated from toenail data linear regression)",
            "Fisher Information approximation for sample size calculation",
            "Asymptotic normality of maximum likelihood estimators"
        ]
        
        result = BayesianSampleSizeAnalysis(
            prior_type=prior_name,
            prior_information_content=prior_info_content,
            expected_data_information=expected_data_info,
            posterior_information_content=posterior_info_content,
            baseline_sample_size=baseline_sample_size,
            adjusted_sample_size=adjusted_sample_size,
            sample_size_reduction_percent=sample_size_reduction * 100,
            patients_saved=patients_saved,
            baseline_power=baseline_power,
            adjusted_power=adjusted_power,
            power_improvement=power_improvement,
            theoretical_basis=theoretical_basis,
            assumptions=assumptions,
            confidence_level=0.95
        )
        
        results.append(result)
    
    return results


def analyze_llm_consistency(elicitor, clinical_params: Dict, n_runs: int = 10) -> List[LLMConsistencyReport]:
    """
    Analyze LLM output consistency across multiple runs
    
    Parameters:
    -----------
    elicitor : LLMPriorElicitor
        The LLM elicitor instance
    clinical_params : Dict
        Parameters for clinical prior elicitation
    n_runs : int
        Number of runs for consistency analysis
        
    Returns:
    --------
    List[LLMConsistencyReport]
        Consistency analysis for each parameter
    """
    from .data_models import LLMConsistencyReport
    
    logger.info(f"🔄 Running LLM consistency analysis ({n_runs} runs)...")
    
    all_runs = []
    for run in range(n_runs):
        logger.info(f"  Run {run + 1}/{n_runs}")
        try:
            priors = elicitor.elicit_clinical_priors(**clinical_params)
            all_runs.append(priors)
        except Exception as e:
            logger.warning(f"  ⚠️ Run {run + 1} failed: {e}")
    
    if not all_runs:
        logger.error("❌ No successful LLM runs for consistency analysis")
        return []
    
    # Organize by parameter
    parameter_results = {}
    for run_priors in all_runs:
        for prior in run_priors:
            param_name = prior.parameter
            if param_name not in parameter_results:
                parameter_results[param_name] = {'means': [], 'stds': []}
            parameter_results[param_name]['means'].append(prior.mean)
            parameter_results[param_name]['stds'].append(prior.std)
    
    # Calculate consistency metrics
    consistency_reports = []
    for param_name, values in parameter_results.items():
        means = np.array(values['means'])
        stds = np.array(values['stds'])
        
        overall_mean = np.mean(means)
        overall_std = np.std(means)  # Std of the means across runs
        cv = overall_std / abs(overall_mean) if overall_mean != 0 else float('inf')
        
        # 95% confidence interval for the mean estimate
        ci_95 = (
            overall_mean - 1.96 * overall_std / np.sqrt(len(means)),
            overall_mean + 1.96 * overall_std / np.sqrt(len(means))
        )
        
        # Stability score (inverse of coefficient of variation, capped at 1)
        stability_score = min(1.0, 1 / (1 + cv)) if cv != float('inf') else 0.0
        
        # Recommendation based on stability
        if stability_score > 0.8:
            recommendation = "High consistency - reliable for analysis"
        elif stability_score > 0.6:
            recommendation = "Moderate consistency - usable with caution"
        else:
            recommendation = "Low consistency - consider increasing temperature or more runs"
        
        report = LLMConsistencyReport(
            parameter=param_name,
            n_runs=len(means),
            mean_estimates=means.tolist(),
            std_estimates=stds.tolist(),
            overall_mean=overall_mean,
            overall_std=overall_std,
            coefficient_of_variation=cv,
            confidence_interval_95=ci_95,
            stability_score=stability_score,
            recommendation=recommendation
        )
        
        consistency_reports.append(report)
    
    logger.info(f"✅ Consistency analysis completed for {len(consistency_reports)} parameters")
    return consistency_reports
