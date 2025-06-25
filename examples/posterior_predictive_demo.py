# filepath: examples/posterior_predictive_demo.py
"""
Posterior Predictive Performance Evaluation Demo
ミーティングで提案された内容の実装例
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    PosteriorPredictiveEvaluator, 
    generate_synthetic_test_data,
    load_historical_priors
)

def main():
    """
    Posterior Predictive Performance Evaluation デモ
    
    実装内容:
    1. 事前分布の分散(σ)を変化させて性能評価
    2. サンプルサイズ vs 予測性能のプロット
    3. 最適なσ値の探索
    """
    print("🔬 Posterior Predictive Performance Evaluation Demo")
    print("="*60)
    
    # 1. テストデータの生成
    print("\n📊 Generating synthetic test data...")
    test_data = generate_synthetic_test_data(n_test=200)
    print(f"✓ Generated {len(test_data)} test observations")
    
    # 2. 事前分布設定の取得
    print("\n📋 Loading prior configurations...")
    historical_priors = load_historical_priors()
    
    # LLM事前分布（現在の設定）
    llm_priors = {
        'beta_intercept': {'dist': 'normal', 'mu': 2.5, 'sigma': 1.0},
        'beta_time': {'dist': 'normal', 'mu': 0.6, 'sigma': 0.2},
        'beta_interaction': {'dist': 'normal', 'mu': 0.0, 'sigma': 0.15}
    }
    
    comparison_setup = {
        'historical_fixed': historical_priors['fixed_effect_model'],
        'llm_expert': llm_priors,
        'uninformative': {
            'beta_intercept': {'dist': 'normal', 'mu': 0, 'sigma': 100},
            'beta_time': {'dist': 'normal', 'mu': 0, 'sigma': 100},
            'beta_interaction': {'dist': 'normal', 'mu': 0, 'sigma': 100}
        }
    }
    
    # 3. Posterior Predictive Evaluatorの初期化
    evaluator = PosteriorPredictiveEvaluator(baseline_sample_size=400)
    
    # 4. サンプルサイズ vs 性能の評価
    print("\n🎯 Evaluating performance across sample sizes...")
    sample_sizes = [50, 100, 150, 200, 250, 300, 350, 400]
    
    performance_results = evaluator.evaluate_prior_performance(
        comparison_setup, 
        test_data, 
        sample_sizes=sample_sizes
    )
    
    print(f"✓ Completed {len(performance_results)} evaluations")
    
    # 結果表示
    print("\n📈 Performance Results (Top 10):")
    print("-" * 80)
    print(f"{'Prior Type':<20} {'σ':<8} {'n':<6} {'Log-Like':<12} {'Coverage':<10} {'RMSE':<8}")
    print("-" * 80)
    
    for result in performance_results[:10]:
        print(f"{result.prior_type:<20} {result.sigma_value:<8.3f} {result.sample_size:<6} "
              f"{result.predictive_log_likelihood:<12.2f} {result.coverage_probability:<10.3f} "
              f"{result.rmse:<8.3f}")
    
    # 5. プロット作成
    print("\n📊 Creating performance plots...")
    
    # サンプルサイズ vs 性能のプロット
    evaluator.plot_performance_vs_sample_size(
        performance_results,
        save_path="results/posterior_predictive_performance_vs_n.png"
    )
    
    # 6. σ値の最適化実験
    print("\n🔍 Optimizing sigma values...")
    
    # treatment_advantage (β_interaction) のσを変化させて実験
    sigma_values = np.logspace(-2, 1, 15)  # 0.01 to 10
    
    benchmark_result = evaluator.benchmark_sigma_values(
        comparison_setup,
        test_data,
        parameter='beta_interaction',
        sigma_values=sigma_values
    )
    
    print(f"\n🏆 Best performing sigma values for beta_interaction:")
    print(f"Best: {benchmark_result.best_performing_prior}")
    print(f"Top 5: {', '.join(benchmark_result.performance_ranking)}")
    
    # σ最適化プロット
    evaluator.plot_sigma_optimization(
        benchmark_result,
        save_path="results/sigma_optimization_beta_interaction.png"
    )
    
    # 7. 要約レポート
    print("\n📋 Summary Report:")
    print("="*60)
    
    # 最適なサンプルサイズでの比較
    best_n_results = [r for r in performance_results if r.sample_size == 400]
    best_n_results.sort(key=lambda x: x.predictive_log_likelihood, reverse=True)
    
    print(f"\nAt n=400 (baseline sample size):")
    for i, result in enumerate(best_n_results[:3]):
        print(f"{i+1}. {result.prior_type} (σ={result.sigma_value:.3f}): "
              f"Log-Likelihood = {result.predictive_log_likelihood:.2f}")
    
    # 現在の設定の問題点
    llm_result = next((r for r in best_n_results if r.prior_type == 'llm_expert'), None)
    if llm_result:
        print(f"\n💡 Current LLM Prior Analysis:")
        print(f"   - Current σ for treatment effect: {llm_result.sigma_value:.3f}")
        print(f"   - Performance rank: {llm_result.rank} out of {len(best_n_results)}")
        print(f"   - Coverage probability: {llm_result.coverage_probability:.3f}")
        
        if llm_result.coverage_probability < 0.90:
            print(f"   ⚠️  Coverage is low - prior might be too narrow!")
        elif llm_result.coverage_probability > 0.98:
            print(f"   ⚠️  Coverage is too high - prior might be too wide!")
        else:
            print(f"   ✓ Coverage looks reasonable")
    
    print(f"\n🎯 Recommendations:")
    print(f"   1. Consider testing wider sigma values (current: 0.15-1.0)")
    print(f"   2. Optimal sigma for treatment effect appears to be: {benchmark_result.best_performing_prior}")
    print(f"   3. Use posterior predictive checks to validate prior choices")
    print(f"   4. Consider adaptive prior selection based on data characteristics")
    
    print(f"\n✓ Demo completed! Check results/ folder for plots.")

if __name__ == "__main__":
    # 結果ディレクトリを作成
    os.makedirs("results", exist_ok=True)
    main()
