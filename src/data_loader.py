# filepath: src/data_loader.py
"""
Data Loading Functions
データ読み込み機能
"""

import os
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def load_historical_priors() -> Dict:
    """
    以前の研究で使用された事前分布を読み込み
    data/Study2-Fixed-Mixed-Effect.Rmdの事前分布パラメータを使用
    
    Fixed Effect Model:
    - beta0 = c(2.5, 0.6, 0)
    - sigma0 = diag(rep(100, 3)) -> std = sqrt(100) = 10
    - nu0 = 2, s20 = 3.7 -> Inverse Gamma(nu0/2, nu0*s20/2) = IG(1, 3.7)
    
    Mixed Effect Model:  
    - mu0 = c(2.5, 0.6, 0)
    - Sigma0 = diag(rep(100, 3)) -> std = 10
    - nu0 = 1, delta0 = 3.7 -> Inverse Gamma(nu0/2, nu0*delta0/2) = IG(0.5, 1.85)
    """
    historical_priors = {
        'fixed_effect_model': {
            'beta_intercept': {'dist': 'normal', 'mu': 2.5, 'sigma': 10.0},
            'beta_time': {'dist': 'normal', 'mu': 0.6, 'sigma': 10.0},
            'beta_interaction': {'dist': 'normal', 'mu': 0.0, 'sigma': 10.0},
            'sigma': {'dist': 'inverse_gamma', 'alpha': 1.0, 'beta': 3.7},  # 修正: nu0*s20/2 = 2*3.7/2 = 3.7
            'source': 'Study2-Fixed-Mixed-Effect.Rmd',
            'reference': 'Fixed Effect Model - nu0=2, s20=3.7'
        },
        'mixed_effect_model': {
            'beta_intercept': {'dist': 'normal', 'mu': 2.5, 'sigma': 10.0},
            'beta_time': {'dist': 'normal', 'mu': 0.6, 'sigma': 10.0}, 
            'beta_interaction': {'dist': 'normal', 'mu': 0.0, 'sigma': 10.0},
            'sigma': {'dist': 'inverse_gamma', 'alpha': 0.5, 'beta': 1.85},  # 修正: nu0*delta0/2 = 1*3.7/2 = 1.85
            'source': 'Study2-Fixed-Mixed-Effect.Rmd',
            'reference': 'Mixed Effect Model - nu0=1, delta0=3.7'
        }
    }
    return historical_priors


def load_actual_toenail_data() -> pd.DataFrame:
    """
    実際の爪真菌症データを読み込み
    """
    try:
        # 現在のスクリプトからの相対パスを使用
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'data', 'toenail.txt')
        
        toenail_data = pd.read_csv(data_path, sep='\s+')
        logger.info(f"✓ Loaded toenail data: {toenail_data.shape[0]} observations, {toenail_data.shape[1]} variables")
        return toenail_data
    except Exception as e:
        logger.error(f"❌ Failed to load toenail data: {e}")
        return pd.DataFrame()
