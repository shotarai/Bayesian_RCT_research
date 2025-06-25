# filepath: src/posterior_predictive_evaluation.py
"""
Posterior Predictive Performance Evaluation System
事前分布の性能をPosterior Predictive Checkで評価
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class PosteriorPredictiveResult:
    """Posterior predictive evaluation result"""
    prior_type: str
    sigma_value: float
    sample_size: int
    predictive_log_likelihood: float
    coverage_probability: float
    prediction_interval_width: float
    bias: float
    rmse: float
    rank: int

@dataclass
class BenchmarkingResult:
    """Benchmarking comparison result"""
    benchmark_id: str
    timestamp: str
    test_data_size: int
    results: List[PosteriorPredictiveResult]
    best_performing_prior: str
    performance_ranking: List[str]

class PosteriorPredictiveEvaluator:
    """
    事前分布の性能をPosterior Predictive Checkで評価するシステム
    
    キーワード対応:
    - Benchmarking: 複数の事前分布設定を系統的に比較
    - P(σ,D): 事前分布σとデータDに基づく予測性能評価
    - posterior predictive performance: ΣP(yi | xi, D(test data))
    """
    
    def __init__(self, baseline_sample_size: int = 400):
        self.baseline_sample_size = baseline_sample_size
        self.sigma_range = np.logspace(-1, 2, 20)  # 0.1 to 100
        self.sample_size_range = np.arange(50, 500, 50)
        
    def evaluate_prior_performance(self, 
                                 priors_config: Dict,
                                 test_data: np.ndarray,
                                 sample_sizes: Optional[List[int]] = None) -> List[PosteriorPredictiveResult]:
        """
        事前分布の性能をサンプルサイズごとに評価
        
        Parameters:
        -----------
        priors_config : Dict
            事前分布設定
        test_data : np.ndarray
            テストデータ（真の分布からのサンプル）
        sample_sizes : List[int]
            評価するサンプルサイズのリスト
            
        Returns:
        --------
        List[PosteriorPredictiveResult]
            各設定での予測性能結果
        """
        if sample_sizes is None:
            sample_sizes = self.sample_size_range
            
        results = []
        
        for prior_type, prior_params in priors_config.items():
            if prior_type == 'uninformative':
                continue
                
            sigma_val = prior_params.get('beta_interaction', {}).get('sigma', 1.0)
            
            for n in sample_sizes:
                # Simulate posterior predictive distribution
                pred_performance = self._calculate_predictive_performance(
                    prior_params, test_data, n
                )
                
                result = PosteriorPredictiveResult(
                    prior_type=prior_type,
                    sigma_value=sigma_val,
                    sample_size=n,
                    predictive_log_likelihood=pred_performance['log_likelihood'],
                    coverage_probability=pred_performance['coverage'],
                    prediction_interval_width=pred_performance['interval_width'],
                    bias=pred_performance['bias'],
                    rmse=pred_performance['rmse'],
                    rank=0  # Will be set later in ranking
                )
                results.append(result)
                
        # Rank results by predictive performance
        results.sort(key=lambda x: x.predictive_log_likelihood, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
            
        return results
    
    def _calculate_predictive_performance(self, 
                                        prior_params: Dict, 
                                        test_data: np.ndarray, 
                                        sample_size: int) -> Dict:
        """
        予測性能指標を計算
        
        Implements:
        - P(σ,D): Posterior predictive likelihood
        - Coverage probability: 95%信頼区間のカバレッジ率
        - Prediction interval width: 予測区間の幅
        """
        # Simulate training data
        np.random.seed(42)  # For reproducibility
        X_train = self._generate_design_matrix(sample_size)
        
        # True parameters (based on actual toenail data)
        true_beta = np.array([1.89, 0.558, 0.042])  # baseline, time, treatment
        true_sigma = 4.39
        
        # Generate training observations
        y_train = X_train @ true_beta + np.random.normal(0, true_sigma, sample_size)
        
        # Calculate posterior parameters
        posterior_mean, posterior_cov = self._calculate_posterior(
            X_train, y_train, prior_params
        )
        
        # Generate test design matrix
        X_test = self._generate_design_matrix(len(test_data))
        
        # Posterior predictive mean and variance
        pred_mean = X_test @ posterior_mean
        pred_var = np.diag(X_test @ posterior_cov @ X_test.T) + true_sigma**2
        
        # Calculate performance metrics
        log_likelihood = np.sum(stats.norm.logpdf(test_data, pred_mean, np.sqrt(pred_var)))
        
        # Coverage probability (95% intervals)
        lower_bound = pred_mean - 1.96 * np.sqrt(pred_var)
        upper_bound = pred_mean + 1.96 * np.sqrt(pred_var)
        coverage = np.mean((test_data >= lower_bound) & (test_data <= upper_bound))
        
        # Prediction interval width
        interval_width = np.mean(upper_bound - lower_bound)
        
        # Bias and RMSE
        bias = np.mean(pred_mean - test_data)
        rmse = np.sqrt(np.mean((pred_mean - test_data)**2))
        
        return {
            'log_likelihood': log_likelihood,
            'coverage': coverage,
            'interval_width': interval_width,
            'bias': bias,
            'rmse': rmse
        }
    
    def _generate_design_matrix(self, n_obs: int) -> np.ndarray:
        """
        線形混合効果モデルの設計行列を生成
        y_ij = β₀ + β₁*time_ij + β₂*treat_i + ε_ij
        """
        # Simulate time points and treatment assignments
        times = np.random.choice([0, 1, 2, 3, 6, 9, 12], n_obs)
        treatments = np.random.choice([0, 1], n_obs)
        
        X = np.column_stack([
            np.ones(n_obs),    # intercept
            times,             # time effect
            treatments         # treatment effect
        ])
        
        return X
    
    def _calculate_posterior(self, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           prior_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        ベイジアン線形回帰の事後分布を計算
        """
        # Prior parameters
        prior_mean = np.array([
            prior_params.get('beta_intercept', {}).get('mu', 0),
            prior_params.get('beta_time', {}).get('mu', 0),
            prior_params.get('beta_interaction', {}).get('mu', 0)
        ])
        
        prior_prec = np.diag([
            1 / prior_params.get('beta_intercept', {}).get('sigma', 100)**2,
            1 / prior_params.get('beta_time', {}).get('sigma', 100)**2,
            1 / prior_params.get('beta_interaction', {}).get('sigma', 100)**2
        ])
        
        # Likelihood precision
        sigma_y = 4.39  # Known error standard deviation
        likelihood_prec = (X.T @ X) / sigma_y**2
        
        # Posterior parameters
        posterior_prec = prior_prec + likelihood_prec
        posterior_cov = np.linalg.inv(posterior_prec)
        posterior_mean = posterior_cov @ (prior_prec @ prior_mean + (X.T @ y) / sigma_y**2)
        
        return posterior_mean, posterior_cov
    
    def benchmark_sigma_values(self, 
                             base_prior_config: Dict,
                             test_data: np.ndarray,
                             parameter: str = 'beta_interaction',
                             sigma_values: Optional[np.ndarray] = None) -> BenchmarkingResult:
        """
        異なるσ値での事前分布性能をベンチマーク
        
        ミーティングで提案された「事前分布の分散を広げたり狭めたりして効果を測定」に対応
        """
        if sigma_values is None:
            sigma_values = self.sigma_range
            
        results = []
        
        for sigma in sigma_values:
            # Create modified prior config
            modified_config = base_prior_config.copy()
            if parameter in modified_config.get('llm_expert', {}):
                modified_config['llm_expert'][parameter]['sigma'] = sigma
                
                # Evaluate performance
                performance_results = self.evaluate_prior_performance(
                    {'modified_prior': modified_config['llm_expert']},
                    test_data,
                    sample_sizes=[self.baseline_sample_size]
                )
                
                if performance_results:
                    result = performance_results[0]
                    result.sigma_value = sigma
                    results.append(result)
        
        # Rank by performance
        results.sort(key=lambda x: x.predictive_log_likelihood, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return BenchmarkingResult(
            benchmark_id=f"sigma_benchmark_{parameter}",
            timestamp=pd.Timestamp.now().isoformat(),
            test_data_size=len(test_data),
            results=results,
            best_performing_prior=f"sigma={results[0].sigma_value:.3f}" if results else "None",
            performance_ranking=[f"σ={r.sigma_value:.3f}" for r in results[:5]]
        )
    
    def plot_performance_vs_sample_size(self, 
                                      results: List[PosteriorPredictiveResult],
                                      save_path: Optional[str] = None) -> None:
        """
        縦軸: Posterior Predictive Performance
        横軸: サンプル数
        のプロットを作成（ミーティング提案に対応）
        """
        plt.figure(figsize=(12, 8))
        
        # Group by prior type
        prior_types = list(set(r.prior_type for r in results))
        
        for prior_type in prior_types:
            type_results = [r for r in results if r.prior_type == prior_type]
            sample_sizes = [r.sample_size for r in type_results]
            performances = [r.predictive_log_likelihood for r in type_results]
            
            plt.plot(sample_sizes, performances, 'o-', label=f'{prior_type}', linewidth=2)
        
        plt.xlabel('Sample Size (n)', fontsize=12)
        plt.ylabel('Posterior Predictive Log-Likelihood', fontsize=12)
        plt.title('Prior Performance vs Sample Size\n(Higher is Better)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_sigma_optimization(self, 
                              benchmark_result: BenchmarkingResult,
                              save_path: Optional[str] = None) -> None:
        """
        σ値の最適化結果をプロット
        """
        plt.figure(figsize=(10, 6))
        
        sigma_values = [r.sigma_value for r in benchmark_result.results]
        performances = [r.predictive_log_likelihood for r in benchmark_result.results]
        
        plt.semilogx(sigma_values, performances, 'bo-', linewidth=2, markersize=6)
        
        # Mark the best performing sigma
        best_idx = np.argmax(performances)
        plt.semilogx(sigma_values[best_idx], performances[best_idx], 
                    'ro', markersize=10, label=f'Best: σ={sigma_values[best_idx]:.3f}')
        
        plt.xlabel('Prior Standard Deviation (σ)', fontsize=12)
        plt.ylabel('Posterior Predictive Log-Likelihood', fontsize=12)
        plt.title('Prior Width Optimization\n(Tradeoff between Informativeness and Flexibility)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sigma optimization plot saved to {save_path}")
        
        plt.show()

def generate_synthetic_test_data(n_test: int = 100) -> np.ndarray:
    """
    テスト用の合成データを生成
    """
    np.random.seed(123)  # For reproducibility
    
    # True parameters from toenail data
    true_beta = np.array([1.89, 0.558, 0.042])
    true_sigma = 4.39
    
    # Generate test design matrix
    times = np.random.choice([0, 1, 2, 3, 6, 9, 12], n_test)
    treatments = np.random.choice([0, 1], n_test)
    
    X_test = np.column_stack([
        np.ones(n_test),
        times,
        treatments
    ])
    
    # Generate test observations
    y_test = X_test @ true_beta + np.random.normal(0, true_sigma, n_test)
    
    return y_test
