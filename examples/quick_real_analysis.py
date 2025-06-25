# filepath: examples/quick_real_analysis.py
"""
Quick Real Analysis with Posterior Predictive Performance
素早く実行可能な実際の分析
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    PosteriorPredictiveEvaluator, 
    generate_synthetic_test_data,
    load_historical_priors,
    MockLLMPriorElicitor
)

def run_quick_analysis():
    """
    クイック分析の実行
    """
    print("🚀 Quick Posterior Predictive Performance Analysis")
    print("=" * 60)
    
    # 1. 事前分布の準備
    print("\n📋 Preparing Prior Configurations...")
    
    # Mock LLM priors
    mock_elicitor = MockLLMPriorElicitor()
    mock_priors = mock_elicitor.elicit_clinical_priors(
        treatment_1="Itraconazole 250mg daily",
        treatment_2="Terbinafine 250mg daily", 
        outcome_measure="unaffected nail length",
        clinical_context="toenail fungal infection"
    )
    mock_analysis_priors = mock_elicitor.export_priors_for_analysis(mock_priors)
    
    # Historical priors
    historical_priors = load_historical_priors()
    
    # Configuration
    priors_config = {
        'mock_llm': mock_analysis_priors,
        'historical_fixed': historical_priors['fixed_effect_model'],
        'uninformative': {
            'beta_intercept': {'dist': 'normal', 'mu': 0, 'sigma': 100},
            'beta_time': {'dist': 'normal', 'mu': 0, 'sigma': 100},
            'beta_interaction': {'dist': 'normal', 'mu': 0, 'sigma': 100}
        }
    }
    
    print("✅ Prior configurations prepared")
    
    # Current settings
    print("\n📊 Current Prior Settings:")
    print("-" * 40)
    for param in ['beta_intercept', 'beta_time', 'beta_interaction']:
        if param in mock_analysis_priors:
            sigma = mock_analysis_priors[param]['sigma']
            mu = mock_analysis_priors[param]['mu']
            print(f"{param:18s}: μ={mu:6.3f}, σ={sigma:6.3f}")
    
    # 2. テストデータ生成
    print("\n🎯 Generating Test Data...")
    test_data = generate_synthetic_test_data(n_test=200)
    print(f"✅ Generated {len(test_data)} test observations")
    
    # 3. 性能評価
    print("\n📈 Performance Evaluation...")
    evaluator = PosteriorPredictiveEvaluator(baseline_sample_size=400)
    
    sample_sizes = [100, 200, 300, 400, 500]
    performance_results = evaluator.evaluate_prior_performance(
        priors_config, test_data, sample_sizes=sample_sizes
    )
    
    print(f"✅ Completed {len(performance_results)} evaluations")
    
    # 4. 結果表示
    print(f"\n📊 Performance Results at n=400:")
    print("-" * 60)
    print(f"{'Prior Type':<18s} {'Log-Like':<10s} {'Coverage':<10s} {'RMSE':<8s} {'σ_treat':<8s}")
    print("-" * 60)
    
    n400_results = [r for r in performance_results if r.sample_size == 400]
    n400_results.sort(key=lambda x: x.predictive_log_likelihood, reverse=True)
    
    for result in n400_results:
        print(f"{result.prior_type:<18s} {result.predictive_log_likelihood:<10.2f} "
              f"{result.coverage_probability:<10.3f} {result.rmse:<8.3f} {result.sigma_value:<8.3f}")
    
    # 5. σ最適化
    print(f"\n🔍 Sigma Optimization for Treatment Effect...")
    sigma_values = np.logspace(-2, 1, 15)  # 0.01 to 10
    
    benchmark_result = evaluator.benchmark_sigma_values(
        priors_config, test_data, 
        parameter='beta_interaction',
        sigma_values=sigma_values
    )
    
    print(f"✅ Best performing σ: {benchmark_result.best_performing_prior}")
    print(f"✅ Top 5 σ values: {', '.join(benchmark_result.performance_ranking[:5])}")
    
    # 6. 可視化
    print(f"\n📊 Creating Visualizations...")
    
    # Performance vs Sample Size
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    prior_types = list(set(r.prior_type for r in performance_results))
    for prior_type in prior_types:
        type_results = [r for r in performance_results if r.prior_type == prior_type]
        sample_sizes_plot = [r.sample_size for r in type_results]
        performances = [r.predictive_log_likelihood for r in type_results]
        plt.plot(sample_sizes_plot, performances, 'o-', label=prior_type, linewidth=2)
    
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Log-Likelihood')
    plt.title('Performance vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Coverage
    plt.subplot(1, 3, 2)
    for prior_type in prior_types:
        type_results = [r for r in performance_results if r.prior_type == prior_type]
        sample_sizes_plot = [r.sample_size for r in type_results]
        coverages = [r.coverage_probability for r in type_results]
        plt.plot(sample_sizes_plot, coverages, 'o-', label=prior_type, linewidth=2)
    
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target 95%')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Coverage Probability')
    plt.title('Coverage vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.85, 1.0)
    
    # Sigma optimization
    plt.subplot(1, 3, 3)
    sigma_vals = [r.sigma_value for r in benchmark_result.results]
    performances_sigma = [r.predictive_log_likelihood for r in benchmark_result.results]
    
    plt.semilogx(sigma_vals, performances_sigma, 'bo-', linewidth=2, markersize=6)
    
    best_idx = np.argmax(performances_sigma)
    plt.semilogx(sigma_vals[best_idx], performances_sigma[best_idx], 
                'ro', markersize=10, label=f'Best: σ={sigma_vals[best_idx]:.3f}')
    
    plt.xlabel('Prior Standard Deviation (σ)')
    plt.ylabel('Log-Likelihood')
    plt.title('Treatment Effect σ Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/quick_analysis_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved: results/quick_analysis_results.png")
    
    # 7. 分析結果のまとめ
    current_sigma = mock_analysis_priors['beta_interaction']['sigma']
    optimal_results = max(benchmark_result.results, key=lambda x: x.predictive_log_likelihood)
    optimal_sigma = optimal_results.sigma_value
    improvement_ratio = optimal_sigma / current_sigma
    
    summary = f"""
# Quick Posterior Predictive Performance Analysis Results

**Analysis Date**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## 🎯 Key Findings

### Current vs Optimal Settings (Treatment Effect)
- **Current σ**: {current_sigma:.3f}
- **Optimal σ**: {optimal_sigma:.3f}  
- **Improvement Factor**: {improvement_ratio:.1f}x {'larger' if improvement_ratio > 1 else 'smaller'}

### Performance Comparison (n=400)
"""
    
    for i, result in enumerate(n400_results):
        rank = i + 1
        status = "🏆 Best" if rank == 1 else f"#{rank}"
        summary += f"""
{status} **{result.prior_type}**:
- Log-Likelihood: {result.predictive_log_likelihood:.2f}
- Coverage: {result.coverage_probability:.1%} {'✅ Good' if 0.93 <= result.coverage_probability <= 0.97 else '⚠️ Needs attention'}
- RMSE: {result.rmse:.3f}
- Treatment σ: {result.sigma_value:.3f}
"""
    
    summary += f"""
### Sigma Optimization Results
- **Best σ for treatment effect**: {optimal_sigma:.3f}
- **Performance improvement**: {optimal_results.predictive_log_likelihood:.2f} vs {performance_results[0].predictive_log_likelihood:.2f}
- **Coverage improvement**: {optimal_results.coverage_probability:.1%}

## 🎯 Recommendations

### Immediate Action
```python
# In llm_elicitor.py, change:
std=0.150  # current treatment_advantage setting

# To:
std={optimal_sigma:.3f}  # optimized setting
```

### Expected Benefits
- Better calibrated uncertainty estimates
- Improved coverage probability (closer to 95%)
- More robust posterior predictions
- Enhanced clinical credibility

---
*Generated by Quick Posterior Predictive Performance Analysis*
"""
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"results/quick_analysis_summary_{timestamp}.md"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✅ Summary report saved: {summary_filename}")
    
    return summary, n400_results, benchmark_result

if __name__ == "__main__":
    summary, performance_results, optimization_result = run_quick_analysis()
    
    print("\n" + "="*70)
    print("📋 COPYABLE ANALYSIS SUMMARY")
    print("="*70)
    print(summary)
