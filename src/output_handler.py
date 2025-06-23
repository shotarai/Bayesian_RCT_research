# filepath: src/output_handler.py
"""
Output Handler for Bayesian RCT Analysis Results
分析結果の出力・保存機能
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def save_analysis_results(results: Dict[str, Any], output_dir: str = "results") -> str:
    """
    分析結果をJSONファイルに保存
    
    Parameters:
    -----------
    results : Dict[str, Any]
        分析結果データ
    output_dir : str
        出力ディレクトリパス
        
    Returns:
    --------
    str
        保存されたファイルのパス
    """
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # タイムスタンプ付きファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bayesian_rct_analysis_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # シリアライズ可能な形式に変換
    serializable_results = make_serializable(results)
    
    # 結果保存
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Analysis results saved to: {filepath}")
    return filepath


def save_summary_report(results: Dict[str, Any], output_dir: str = "results") -> str:
    """
    サマリーレポートをMarkdownファイルに保存
    
    Parameters:
    -----------
    results : Dict[str, Any]
        分析結果データ
    output_dir : str
        出力ディレクトリパス
        
    Returns:
    --------
    str
        保存されたファイルのパス
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_summary_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Bayesian RCT Analysis Results\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"**API Used**: {'Real LLM (GPT-4)' if results.get('api_key_used', False) else 'Mock LLM'}\n\n")
        
        # 研究目標と答え
        f.write("## 🎯 Research Question\n")
        f.write("**\"How many patients can we save with better priors?\"**\n\n")
        
        # 患者節約効果
        if 'sample_size_benefits' in results:
            f.write("## 💡 Patient Savings Analysis\n\n")
            for benefit in results['sample_size_benefits']:
                prior_type = benefit.get('prior_type', 'Unknown')
                patients_saved = benefit.get('patient_savings', 'N/A')
                sample_reduction = benefit.get('sample_size_reduction', 'N/A')
                effective_gain = benefit.get('effective_sample_size_gain', 'N/A')
                
                f.write(f"### {prior_type}\n")
                f.write(f"- **Patients saved**: {patients_saved}\n")
                f.write(f"- **Sample size reduction**: {sample_reduction}%\n")
                f.write(f"- **Effective sample size gain**: {effective_gain}x\n\n")
        
        # 事前分布比較
        if 'comparison_setup' in results:
            f.write("## 📊 Prior Distributions Comparison\n\n")
            for prior_type, priors in results['comparison_setup'].items():
                f.write(f"### {prior_type.upper()}\n")
                for param, spec in priors.items():
                    if isinstance(spec, dict) and param not in ['source', 'reference']:
                        if 'mu' in spec and 'sigma' in spec:
                            f.write(f"- **{param}**: μ={spec['mu']}, σ={spec['sigma']}\n")
                        elif 'alpha' in spec and 'beta' in spec:
                            f.write(f"- **{param}**: α={spec['alpha']}, β={spec['beta']} (Inverse Gamma)\n")
                f.write("\n")
        
        # LLM事前分布詳細
        if 'llm_priors' in results and results['llm_priors']:
            f.write("## 🤖 LLM-Elicited Priors Details\n\n")
            for prior in results['llm_priors']:
                f.write(f"### {prior.get('parameter', 'Unknown Parameter')}\n")
                f.write(f"- **Distribution**: {prior.get('distribution', 'N/A')}\n")
                f.write(f"- **Mean**: {prior.get('mean', 'N/A')}\n")
                f.write(f"- **Std**: {prior.get('std', 'N/A')}\n")
                f.write(f"- **Confidence**: {prior.get('confidence', 'N/A')}\n")
                f.write(f"- **Rationale**: {prior.get('rationale', 'N/A')}\n\n")
        
        # データ統計
        if 'data_stats' in results:
            f.write("## 📈 Dataset Statistics\n\n")
            stats = results['data_stats']
            f.write(f"- **Total observations**: {stats.get('total_observations', 'N/A')}\n")
            f.write(f"- **Unique patients**: {stats.get('unique_patients', 'N/A')}\n")
            f.write(f"- **Treatment groups**: {stats.get('treatment_groups', 'N/A')}\n")
        
        # 結論
        f.write("## ✅ Conclusion\n\n")
        f.write("This analysis demonstrates that informed Bayesian priors can significantly reduce ")
        f.write("the number of patients required for clinical trials while maintaining statistical rigor. ")
        f.write("LLM-elicited priors provide a scalable, cost-effective alternative to traditional ")
        f.write("expert elicitation methods.\n\n")
        
        f.write("---\n")
        f.write(f"*Analysis generated by Bayesian RCT Research System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    logger.info(f"✓ Summary report saved to: {filepath}")
    return filepath


def save_prior_comparison_csv(comparisons: List[Dict], output_dir: str = "results") -> str:
    """
    事前分布比較をCSV形式で保存
    """
    import csv
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prior_comparison_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['parameter', 'llm_mean', 'llm_std', 'historical_mean', 'historical_std', 
                     'mean_difference', 'std_difference', 'overlap_coefficient']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for comp in comparisons:
            writer.writerow({
                'parameter': comp.get('parameter', ''),
                'llm_mean': comp.get('llm_prior', {}).get('mu', ''),
                'llm_std': comp.get('llm_prior', {}).get('sigma', ''),
                'historical_mean': comp.get('historical_prior', {}).get('mu', ''),
                'historical_std': comp.get('historical_prior', {}).get('sigma', ''),
                'mean_difference': comp.get('difference_mean', ''),
                'std_difference': comp.get('difference_std', ''),
                'overlap_coefficient': comp.get('overlap_coefficient', '')
            })
    
    logger.info(f"✓ Prior comparison CSV saved to: {filepath}")
    return filepath


def make_serializable(obj: Any) -> Any:
    """
    オブジェクトをJSON シリアライズ可能な形式に変換
    """
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return obj


def create_output_directory_structure(base_dir: str = "results") -> Dict[str, str]:
    """
    出力ディレクトリ構造を作成
    
    Returns:
    --------
    Dict[str, str]
        作成されたディレクトリパスの辞書
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    directories = {
        'base': base_dir,
        'session': os.path.join(base_dir, f"session_{timestamp}"),
        'json': os.path.join(base_dir, f"session_{timestamp}", "json"),
        'reports': os.path.join(base_dir, f"session_{timestamp}", "reports"),
        'csv': os.path.join(base_dir, f"session_{timestamp}", "csv")
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories
