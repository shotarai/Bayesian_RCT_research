# filepath: src/adaptive_prior_selector.py
"""
Adaptive Prior Selection System
ミーティング提案に基づく動的事前分布選択システム
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from .posterior_predictive_evaluation import PosteriorPredictiveEvaluator, generate_synthetic_test_data

logger = logging.getLogger(__name__)

@dataclass
class AdaptivePriorConfig:
    """Adaptive prior configuration result"""
    parameter_name: str
    optimal_sigma: float
    confidence_interval: Tuple[float, float]
    performance_score: float
    recommendation: str
    context_factors: Dict

class AdaptivePriorSelector:
    """
    ミーティング提案の「Notion of goodness of effectsize」および
    「Where does knowledge come from?」に対応する適応的事前分布選択システム
    """
    
    def __init__(self):
        self.evaluator = PosteriorPredictiveEvaluator()
        self.optimization_history = []
        
    def optimize_sigma_for_context(self, 
                                 base_priors: Dict,
                                 sample_size: int,
                                 clinical_domain: str = "onychomycosis",
                                 expected_effect_size: Optional[float] = None) -> Dict[str, AdaptivePriorConfig]:
        """
        文脈に基づく事前分布の最適化
        
        Parameters:
        -----------
        base_priors : Dict
            ベース事前分布設定
        sample_size : int  
            予定サンプルサイズ
        clinical_domain : str
            臨床領域
        expected_effect_size : float
            期待効果サイズ
            
        Returns:
        --------
        Dict[str, AdaptivePriorConfig]
            各パラメータの最適化結果
        """
        logger.info(f"🎯 Optimizing priors for context: {clinical_domain}, n={sample_size}")
        
        # Generate context-appropriate test data
        test_data = self._generate_context_data(clinical_domain, expected_effect_size)
        
        results = {}
        parameters = ['beta_intercept', 'beta_time', 'beta_interaction']
        
        for param in parameters:
            logger.info(f"  Optimizing {param}...")
            
            optimal_config = self._optimize_single_parameter(
                base_priors, param, test_data, sample_size
            )
            results[param] = optimal_config
            
        return results
    
    def _optimize_single_parameter(self, 
                                 base_priors: Dict,
                                 parameter: str,
                                 test_data: np.ndarray,
                                 sample_size: int) -> AdaptivePriorConfig:
        """
        単一パラメータの最適化
        """
        # Sigma values to test (log-spaced)
        sigma_candidates = np.logspace(-2, 1.5, 20)  # 0.01 to ~31.6
        
        performances = []
        
        for sigma in sigma_candidates:
            # Create modified prior config
            modified_priors = self._create_modified_priors(base_priors, parameter, sigma)
            
            # Evaluate performance
            perf_results = self.evaluator.evaluate_prior_performance(
                modified_priors,
                test_data,
                sample_sizes=[sample_size]
            )
            
            if perf_results:
                performances.append({
                    'sigma': sigma,
                    'log_likelihood': perf_results[0].predictive_log_likelihood,
                    'coverage': perf_results[0].coverage_probability,
                    'rmse': perf_results[0].rmse
                })
        
        # Find optimal sigma
        best_perf = max(performances, key=lambda x: x['log_likelihood'])
        optimal_sigma = best_perf['sigma']
        
        # Calculate confidence interval (based on performance stability)
        sorted_perfs = sorted(performances, key=lambda x: x['log_likelihood'], reverse=True)
        top_10_percent = sorted_perfs[:max(1, len(sorted_perfs)//10)]
        sigma_range = [p['sigma'] for p in top_10_percent]
        confidence_interval = (min(sigma_range), max(sigma_range))
        
        # Generate recommendation
        recommendation = self._generate_recommendation(parameter, optimal_sigma, best_perf)
        
        # Context factors
        context_factors = {
            'sample_size': sample_size,
            'n_candidates_tested': len(sigma_candidates),
            'performance_stability': len(top_10_percent),
            'optimal_coverage': best_perf['coverage'],
            'optimal_rmse': best_perf['rmse']
        }
        
        return AdaptivePriorConfig(
            parameter_name=parameter,
            optimal_sigma=optimal_sigma,
            confidence_interval=confidence_interval,
            performance_score=best_perf['log_likelihood'],
            recommendation=recommendation,
            context_factors=context_factors
        )
    
    def _create_modified_priors(self, base_priors: Dict, parameter: str, sigma: float) -> Dict:
        """
        指定パラメータのσを変更した事前分布を作成
        """
        modified = {}
        
        for prior_type, priors in base_priors.items():
            if prior_type == 'uninformative':
                continue
                
            modified_prior = priors.copy()
            if parameter in modified_prior:
                modified_prior[parameter] = modified_prior[parameter].copy()
                modified_prior[parameter]['sigma'] = sigma
                
            modified[f'{prior_type}_modified'] = modified_prior
            
        return modified
    
    def _generate_context_data(self, clinical_domain: str, expected_effect_size: Optional[float]) -> np.ndarray:
        """
        臨床領域に応じたテストデータ生成
        """
        if clinical_domain == "onychomycosis":
            # Use toenail data parameters
            return generate_synthetic_test_data(n_test=200)
        else:
            # Default synthetic data
            return generate_synthetic_test_data(n_test=200)
    
    def _generate_recommendation(self, parameter: str, optimal_sigma: float, performance: Dict) -> str:
        """
        最適化結果に基づく推奨を生成
        """
        coverage = performance['coverage']
        rmse = performance['rmse']
        
        if parameter == 'beta_interaction':  # Treatment effect
            if optimal_sigma < 0.1:
                return f"Very informative prior (σ={optimal_sigma:.3f}). Ensure clinical justification."
            elif optimal_sigma > 10:
                return f"Weakly informative prior (σ={optimal_sigma:.3f}). Consider if more precision is available."
            else:
                return f"Well-calibrated prior (σ={optimal_sigma:.3f}). Good balance of information and flexibility."
        
        elif parameter == 'beta_time':  # Time effect
            if coverage < 0.90:
                return f"Prior may be too narrow (coverage={coverage:.1%}). Consider σ ≥ {optimal_sigma*1.5:.2f}."
            else:
                return f"Appropriate time effect prior (σ={optimal_sigma:.3f})."
        
        else:  # Intercept
            return f"Baseline parameter optimized (σ={optimal_sigma:.3f})."

def demonstrate_adaptive_selection():
    """
    Adaptive Prior Selection のデモンストレーション
    """
    print("🎯 Adaptive Prior Selection Demo")
    print("="*50)
    
    # Initialize selector
    selector = AdaptivePriorSelector()
    
    # Base prior configuration (current LLM settings)
    base_priors = {
        'llm_expert': {
            'beta_intercept': {'dist': 'normal', 'mu': 2.5, 'sigma': 1.0},
            'beta_time': {'dist': 'normal', 'mu': 0.6, 'sigma': 0.2},
            'beta_interaction': {'dist': 'normal', 'mu': 0.0, 'sigma': 0.15}
        }
    }
    
    # Test different scenarios
    scenarios = [
        {'sample_size': 100, 'domain': 'onychomycosis', 'effect_size': 0.05},
        {'sample_size': 400, 'domain': 'onychomycosis', 'effect_size': 0.05},
        {'sample_size': 1000, 'domain': 'onychomycosis', 'effect_size': 0.05}
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n📊 Scenario {i+1}: n={scenario['sample_size']}")
        print("-" * 40)
        
        # Optimize priors for this scenario
        optimal_configs = selector.optimize_sigma_for_context(
            base_priors,
            scenario['sample_size'],
            scenario['domain'],
            scenario['effect_size']
        )
        
        # Display results
        for param, config in optimal_configs.items():
            current_sigma = base_priors['llm_expert'][param]['sigma']
            print(f"{param}:")
            print(f"  Current σ: {current_sigma:.3f}")
            print(f"  Optimal σ: {config.optimal_sigma:.3f}")
            print(f"  95% CI: ({config.confidence_interval[0]:.3f}, {config.confidence_interval[1]:.3f})")
            print(f"  Recommendation: {config.recommendation}")
            print()

if __name__ == "__main__":
    demonstrate_adaptive_selection()
