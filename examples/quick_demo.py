# filepath: examples/quick_demo.py
"""
Quick Demo Script for Bayesian RCT Research
Simple demonstration script
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import MockLLMPriorElicitor, load_historical_priors


def quick_demo():
    """
    5-minute system demonstration
    """
    print("="*60)
    print("🚀 QUICK DEMO: LLM Prior Elicitation System")
    print("="*60)
    
    # Initialize Mock LLM
    print("\n1️⃣ Initialize Mock LLM (no API key needed)")
    elicitor = MockLLMPriorElicitor()
    
    # Set up prior distributions
    print("\n2️⃣ Elicit clinical priors for toenail fungal infection")
    llm_priors = elicitor.elicit_clinical_priors(
        treatment_1="Itraconazole 250mg daily",
        treatment_2="Terbinafine 250mg daily", 
        outcome_measure="unaffected nail length (mm)",
        clinical_context="toenail fungal infection"
    )
    
    # Display results
    print("\n3️⃣ LLM-elicited priors:")
    for prior in llm_priors:
        print(f"   {prior.parameter}: μ={prior.mean:.2f}, σ={prior.std:.2f}")
    
    # Compare with historical priors
    print("\n4️⃣ Compare with historical priors:")
    historical = load_historical_priors()
    hist_fixed = historical['fixed_effect_model']
    
    print("   Historical (fixed model):")
    for param in ['beta_intercept', 'beta_time', 'beta_interaction']:
        if param in hist_fixed:
            spec = hist_fixed[param]
            print(f"   {param}: μ={spec['mu']:.2f}, σ={spec['sigma']:.2f}")
    
    # Export for Bayesian analysis
    print("\n5️⃣ Export for Bayesian analysis:")
    analysis_priors = elicitor.export_priors_for_analysis(llm_priors)
    
    print("   Ready for Bayesian RCT analysis! ✅")
    print(f"   Exported {len(analysis_priors)} parameters")
    
    print("\n" + "="*60)
    print("Demo completed! Run main.py for full analysis.")
    print("="*60)


if __name__ == "__main__":
    quick_demo()
