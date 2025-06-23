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
    load_actual_toenail_data,
    save_analysis_results,
    save_summary_report,
    save_prior_comparison_csv,
    create_output_directory_structure
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_analysis(api_key: Optional[str] = None, save_results: bool = True):
    """
    Execute complete comparative analysis with file output functionality
    
    Parameters:
    -----------
    api_key : Optional[str]
        OpenAI API key (uses Mock LLM if None)
    save_results : bool
        Whether to save results to files
        
    Returns:
    --------
    dict
        Analysis results dictionary
    """
    print("="*80)
    print("COMPLETE BAYESIAN RCT PRIOR ANALYSIS")
    print("LLM vs Historical vs Uninformative Priors")
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
    
    # Sample size reduction effects
    sample_size_benefits = calculate_sample_size_benefits(comparison_setup)
    
    print("\n💡 SAMPLE SIZE REDUCTION ANALYSIS:")
    print("-" * 60)
    for benefit in sample_size_benefits:
        print(f"\n{benefit.prior_type.upper()}:")
        print(f"  Effective sample size gain: {benefit.effective_sample_size:.2f}x")
        print(f"  Sample size reduction: {benefit.sample_size_reduction:.1%}")
        print(f"  Potential patients saved: {benefit.patient_savings}")
        print(f"  Improved power: {benefit.power:.3f}")
    
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
    
    print("\n✅ Complete analysis finished!")
    print("\nSUMMARY:")
    print("1. LLM-elicited priors provide informative alternatives")
    print("2. Sample size reductions possible with informed priors")
    print("3. Patient savings achievable through better prior knowledge")
    
    # Create results data structure
    results = {
        'comparison_setup': comparison_setup,
        'sample_size_benefits': [
            {
                'prior_type': benefit.prior_type if hasattr(benefit, 'prior_type') else str(benefit),
                'patient_savings': benefit.patient_savings if hasattr(benefit, 'patient_savings') else getattr(benefit, 'patients_saved', 'N/A'),
                'sample_size_reduction': benefit.sample_size_reduction if hasattr(benefit, 'sample_size_reduction') else 'N/A',
                'effective_sample_size_gain': benefit.effective_sample_size_gain if hasattr(benefit, 'effective_sample_size_gain') else 'N/A'
            } for benefit in sample_size_benefits
        ],
        'toenail_data': toenail_data.to_dict() if not toenail_data.empty else {},
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
        'data_stats': {
            'total_observations': len(toenail_data) if not toenail_data.empty else 0,
            'unique_patients': toenail_data['id'].nunique() if not toenail_data.empty and 'id' in toenail_data.columns else 0,
            'treatment_groups': toenail_data['treat'].value_counts().to_dict() if not toenail_data.empty and 'treat' in toenail_data.columns else {}
        },
        'analysis_timestamp': datetime.now().isoformat(),
        'api_key_used': api_key is not None
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
        print(f"✓ Sample size benefits analyzed for {len(results['sample_size_benefits'])} prior specifications")
        print(f"✓ Toenail data: {results['data_stats']['total_observations']} observations")
        print(f"✓ LLM priors: {len(results['llm_priors'])} parameters elicited")


if __name__ == "__main__":
    main()
