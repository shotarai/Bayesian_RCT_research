# Complete LLM Prior Elicitation System - Modularized Version
# Modularized LLM Prior Distribution Elicitation System

import os
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Import local modules
from src import (
    comparative_analysis_setup,
    compare_prior_specifications,
    calculate_sample_size_benefits,
    calculate_bayesian_sample_size_analysis,
    analyze_llm_consistency,
    load_actual_toenail_data,
    save_analysis_results,
    save_summary_report,
    save_prior_comparison_csv,
    create_output_directory_structure
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_analysis(api_key: Optional[str] = None, save_results: bool = True, n_llm_runs: int = 5):
    """
    Execute complete comparative analysis with enhanced theoretical rigor
    
    Parameters:
    -----------
    api_key : Optional[str]
        OpenAI API key (uses Mock LLM if None)
    save_results : bool
        Whether to save results to files
    n_llm_runs : int
        Number of LLM runs for consistency analysis
        
    Returns:
    --------
    dict
        Enhanced analysis results dictionary
    """
    print("="*80)
    print("ENHANCED BAYESIAN RCT PRIOR ANALYSIS")
    print("LLM vs Historical vs Uninformative Priors")
    print("With Fisher Information Theory & Consistency Analysis")
    print("="*80)
    
    # Load data
    toenail_data = load_actual_toenail_data()
    
    # Setup comparison
    comparison_setup, llm_priors = comparative_analysis_setup(api_key)
    
    # Prior distributions comparison
    print("\n📊 PRIOR DISTRIBUTIONS COMPARISON:")
    print("-" * 60)
    for prior_type, priors in comparison_setup.items():
        print(f"\n{prior_type.upper()}:")
        for param, spec in priors.items():
            if isinstance(spec, dict) and 'mu' in spec and 'sigma' in spec:
                print(f"  {param}: μ={spec['mu']:.3f}, σ={spec['sigma']:.3f}")
    
    # Enhanced Bayesian sample size analysis (rigorous)
    bayesian_analyses = calculate_bayesian_sample_size_analysis(comparison_setup)
    
    print("\n🔬 ENHANCED BAYESIAN SAMPLE SIZE ANALYSIS:")
    print("-" * 60)
    for analysis in bayesian_analyses:
        print(f"\n{analysis.prior_type.upper()}:")
        print(f"  Prior information content: {analysis.prior_information_content:.3f}")
        print(f"  Expected data information: {analysis.expected_data_information:.3f}")
        print(f"  Sample size reduction: {analysis.sample_size_reduction_percent:.1f}%")
        print(f"  Patients saved: {analysis.patients_saved}")
        print(f"  Power improvement: +{analysis.power_improvement:.3f}")
        print(f"  Theoretical basis: {analysis.theoretical_basis}")
    
    # LLM Consistency Analysis (only for production LLM)
    consistency_reports = []
    if api_key and n_llm_runs > 1:
        print(f"\n🔄 LLM CONSISTENCY ANALYSIS ({n_llm_runs} runs):")
        print("-" * 60)
        
        try:
            from src.llm_elicitor import ProductionLLMPriorElicitor
            elicitor = ProductionLLMPriorElicitor(api_key=api_key)
            
            clinical_params = {
                'treatment_1': "Itraconazole 250mg daily for 12 weeks",
                'treatment_2': "Terbinafine 250mg daily for 12 weeks",
                'outcome_measure': "unaffected nail length in millimeters",
                'clinical_context': "toenail fungal infection (onychomycosis) in adults"
            }
            
            consistency_reports = analyze_llm_consistency(elicitor, clinical_params, n_llm_runs)
            
            for report in consistency_reports:
                print(f"\n{report.parameter}:")
                print(f"  Mean estimate: {report.overall_mean:.3f} ± {report.overall_std:.3f}")
                print(f"  Coefficient of variation: {report.coefficient_of_variation:.3f}")
                print(f"  Stability score: {report.stability_score:.3f}")
                print(f"  Recommendation: {report.recommendation}")
                
        except Exception as e:
            print(f"⚠️ Consistency analysis failed: {e}")
            print("Continuing with single run analysis...")
    else:
        print(f"\n🤖 Mock LLM used - consistency analysis skipped")
        print("(Mock LLM is deterministic)")
    
    # Legacy sample size analysis for comparison
    legacy_benefits = calculate_sample_size_benefits(comparison_setup)
    
    print("\n📊 LEGACY SAMPLE SIZE ANALYSIS (for comparison):")
    print("-" * 60)
    for benefit in legacy_benefits:
        print(f"\n{benefit.prior_type.upper()}:")
        print(f"  Legacy effective sample size gain: {benefit.effective_sample_size:.2f}x")
        print(f"  Legacy sample size reduction: {benefit.sample_size_reduction:.1%}")
        print(f"  Legacy patients saved: {benefit.patient_savings}")
    
    # LLM vs historical priors comparison
    if 'historical_fixed' in comparison_setup and 'llm_expert' in comparison_setup:
        llm_vs_historical = compare_prior_specifications(
            comparison_setup['llm_expert'],
            comparison_setup['historical_fixed']
        )
        
        print("\n🔍 LLM vs HISTORICAL PRIORS COMPARISON:")
        print("-" * 60)
        for comp in llm_vs_historical:
            print(f"\n{comp.parameter}:")
            print(f"  Mean difference: {comp.difference_mean:.3f}")
            print(f"  Std difference: {comp.difference_std:.3f}")
            print(f"  Distribution overlap: {comp.overlap_coefficient:.3f}")
    
    print("\n✅ Enhanced analysis completed!")
    print("\nSUMMARY:")
    print("1. Enhanced theoretical foundation with Fisher Information Theory")
    print("2. LLM consistency analysis for reliability assessment")
    print("3. More conservative and rigorous sample size calculations")
    print("4. Comprehensive theoretical justification provided")
    
    # Create enhanced results data structure
    results = {
        'comparison_setup': comparison_setup,
        'llm_priors': [
            {
                'parameter': prior.parameter if hasattr(prior, 'parameter') else 'unknown',
                'distribution': prior.distribution if hasattr(prior, 'distribution') else 'unknown',
                'mean': prior.mean if hasattr(prior, 'mean') else 'N/A',
                'std': prior.std if hasattr(prior, 'std') else 'N/A',
                'confidence': prior.confidence if hasattr(prior, 'confidence') else 'N/A',
                'rationale': prior.rationale if hasattr(prior, 'rationale') else 'N/A'
            } for prior in llm_priors
        ],
        'enhanced_bayesian_analyses': [
            {
                'prior_type': analysis.prior_type,
                'prior_information_content': analysis.prior_information_content,
                'expected_data_information': analysis.expected_data_information,
                'sample_size_reduction_percent': analysis.sample_size_reduction_percent,
                'patients_saved': analysis.patients_saved,
                'power_improvement': analysis.power_improvement,
                'theoretical_basis': analysis.theoretical_basis,
                'assumptions': analysis.assumptions,
                'confidence_level': analysis.confidence_level
            } for analysis in bayesian_analyses
        ],
        'consistency_reports': [
            {
                'parameter': report.parameter,
                'n_runs': report.n_runs,
                'overall_mean': report.overall_mean,
                'overall_std': report.overall_std,
                'coefficient_of_variation': report.coefficient_of_variation,
                'stability_score': report.stability_score,
                'recommendation': report.recommendation
            } for report in consistency_reports
        ],
        'legacy_sample_size_benefits': [
            {
                'prior_type': benefit.prior_type if hasattr(benefit, 'prior_type') else str(benefit),
                'patient_savings': benefit.patient_savings if hasattr(benefit, 'patient_savings') else getattr(benefit, 'patients_saved', 'N/A'),
                'sample_size_reduction': benefit.sample_size_reduction if hasattr(benefit, 'sample_size_reduction') else 'N/A',
                'effective_sample_size_gain': getattr(benefit, 'effective_sample_size', 'N/A')
            } for benefit in legacy_benefits
        ],
        'toenail_data': toenail_data.to_dict() if not toenail_data.empty else {},
        'data_stats': {
            'total_observations': len(toenail_data) if not toenail_data.empty else 0,
            'unique_patients': toenail_data['id'].nunique() if not toenail_data.empty and 'id' in toenail_data.columns else 0,
            'treatment_groups': toenail_data['treat'].value_counts().to_dict() if not toenail_data.empty and 'treat' in toenail_data.columns else {}
        },
        'analysis_timestamp': datetime.now().isoformat(),
        'n_llm_runs': n_llm_runs,
        'api_key_used': api_key is not None,
        'theoretical_enhancements': {
            'fisher_information_theory': True,
            'consistency_analysis': len(consistency_reports) > 0,
            'rigorous_sample_size_calculation': True,
            'theoretical_validation': True
        }
    }
    
    # File output
    if save_results:
        try:
            print("\n💾 Saving analysis results...")
            
            # JSON detailed results save
            json_path = save_analysis_results(results)
            print(f"📄 Detailed results: {json_path}")
            
            # Markdown summary save
            summary_path = save_summary_report(results)
            print(f"📋 Summary report: {summary_path}")
            
            # Prior distribution comparison CSV save
            prior_comparisons = compare_prior_specifications(
                comparison_setup.get('llm_expert', {}),
                comparison_setup.get('historical_fixed', {})
            )
            if prior_comparisons:
                csv_path = save_prior_comparison_csv([comp.__dict__ if hasattr(comp, '__dict__') else comp for comp in prior_comparisons])
                print(f"📊 Prior comparison CSV: {csv_path}")
            
            print("✅ All results saved successfully!")
            
        except Exception as e:
            print(f"⚠️  Error saving results: {e}")
            print("Analysis completed but files not saved.")
    
    return results


def main():
    """
    Main execution function
    """
    # Get API key from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("ℹ️  OpenAI API key not found - using Mock LLM for demonstration")
        print("To use real LLM, set: export OPENAI_API_KEY='your-api-key'")
        print()
    
    # Execute complete comparative analysis
    print("🚀 Starting complete Bayesian RCT prior analysis...")
    results = run_complete_analysis(api_key)
    
    if results:
        print("\n📈 Analysis completed successfully!")
        print(f"✓ Comparison setup: {len(results['comparison_setup'])} prior types compared")
        print(f"✓ Enhanced Bayesian analyses: {len(results.get('enhanced_bayesian_analyses', []))} specifications")
        print(f"✓ Consistency reports: {len(results.get('consistency_reports', []))} parameters analyzed")
        print(f"✓ Toenail data: {results['data_stats']['total_observations']} observations")
        print(f"✓ LLM priors: {len(results['llm_priors'])} parameters elicited")
        print(f"✓ LLM runs: {results.get('n_llm_runs', 1)} for reliability assessment")


if __name__ == "__main__":
    main()
