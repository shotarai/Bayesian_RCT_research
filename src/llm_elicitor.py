# filepath: src/llm_elicitor.py
"""
LLM Prior Elicitation Classes
LLMベースの事前分布設定システム
"""

import os
import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import asdict

from .data_models import LLMPriorSpecification, ConsistencyAnalysis

logger = logging.getLogger(__name__)


class ProductionLLMPriorElicitor:
    """
    本格的なLLM事前分布設定システム
    OpenAI GPT-4.1 API を使用
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4.1", temperature: float = 0.2):
        """
        初期化
        
        Parameters:
        -----------
        api_key : str
            OpenAI API キー (必須)
        model : str
            使用するモデル（gpt-4.1推奨）
        temperature : float
            応答の一貫性制御（0.1-0.3推奨）
        """
        if not api_key:
            raise ValueError("OpenAI API key is required for production use")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # API接続テスト
        self._test_api_connection()
        
        logger.info(f"✓ LLM Prior Elicitor initialized successfully")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Session ID: {self.session_id}")
    
    def _test_api_connection(self):
        """API接続テスト"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10,
                temperature=0
            )
            logger.info("✓ OpenAI API connection successful")
        except Exception as e:
            logger.error(f"❌ OpenAI API connection failed: {e}")
            raise ConnectionError(f"Cannot connect to OpenAI API: {e}")
    
    def elicit_clinical_priors(self, 
                              treatment_1: str, 
                              treatment_2: str, 
                              outcome_measure: str,
                              clinical_context: str,
                              additional_context: Optional[str] = None) -> List[LLMPriorSpecification]:
        """
        LLMから臨床的事前分布を設定
        """
        logger.info("🔬 Starting LLM-based prior elicitation...")
        logger.info(f"  Context: {clinical_context}")
        logger.info(f"  Comparison: {treatment_2} vs {treatment_1}")
        logger.info(f"  Outcome: {outcome_measure}")
        
        prompt = self._create_expert_prompt(
            treatment_1, treatment_2, outcome_measure, clinical_context, additional_context
        )
        
        priors = self._get_llm_response(prompt)
        
        logger.info(f"✓ Successfully elicited {len(priors)} prior distributions")
        return priors
    
    def _create_expert_prompt(self, treatment_1: str, treatment_2: str, 
                             outcome_measure: str, clinical_context: str,
                             additional_context: Optional[str] = None) -> str:
        """
        専門家相談用の詳細プロンプト作成
        """
        additional_info = f"\n\nADDITIONAL CLINICAL INFORMATION:\n{additional_context}" if additional_context else ""
        
        prompt = f"""
You are a senior clinical biostatistician and medical expert with extensive experience in {clinical_context}. 

CLINICAL STUDY DESIGN:
- Control Treatment: {treatment_1}
- Experimental Treatment: {treatment_2}
- Primary Outcome: {outcome_measure}
- Clinical Context: {clinical_context}
{additional_info}

STATISTICAL MODEL: {outcome_measure} = β₀ + β₁×time + β₂×(time×treatment) + ε

Please provide expert prior beliefs for each parameter in strict JSON format:
{{
    "expert_assessment": {{
        "baseline_intercept": {{"mean": <value>, "std": <value>, "confidence": <0-1>, "rationale": "<reasoning>"}},
        "time_effect": {{"mean": <value>, "std": <value>, "confidence": <0-1>, "rationale": "<reasoning>"}},
        "treatment_advantage": {{"mean": <value>, "std": <value>, "confidence": <0-1>, "rationale": "<reasoning>"}},
        "error_std": {{"mean": <value>, "std": <value>, "confidence": <0-1>, "rationale": "<reasoning>"}}
    }},
    "overall_confidence": <0-1>
}}
"""
        return prompt
    
    def _get_llm_response(self, prompt: str, max_retries: int = 3) -> List[LLMPriorSpecification]:
        """
        実際のLLM API呼び出し（エラーハンドリング付き）
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Consulting LLM expert (attempt {attempt + 1}/{max_retries})...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a clinical biostatistician. Provide precise priors in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content.strip()
                
                # JSON抽出
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                
                llm_data = json.loads(content)
                
                # データ変換
                priors = []
                expert_assessment = llm_data.get("expert_assessment", {})
                
                for param_name, param_data in expert_assessment.items():
                    prior = LLMPriorSpecification(
                        parameter=param_name,
                        distribution="normal",
                        mean=float(param_data["mean"]),
                        std=float(param_data["std"]),
                        confidence=float(param_data["confidence"]),
                        rationale=param_data["rationale"],
                        llm_model=self.model,
                        timestamp=datetime.now().isoformat(),
                        session_id=self.session_id
                    )
                    priors.append(prior)
                
                logger.info(f"✓ LLM consultation successful")
                return priors
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"LLM response could not be parsed as JSON: {e}")
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"⚠️  API call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise ConnectionError(f"LLM API call failed: {e}")
                time.sleep(5)
        
        raise RuntimeError("Failed to get valid LLM response after all retries")
    
    def export_priors_for_analysis(self, priors: List[LLMPriorSpecification]) -> Dict:
        """
        ベイズ解析用の事前分布仕様をエクスポート
        """
        exported_priors = {}
        
        parameter_mapping = {
            'baseline_intercept': 'beta_intercept',
            'time_effect': 'beta_time',
            'treatment_advantage': 'beta_interaction',
            'error_std': 'sigma'
        }
        
        for prior in priors:
            if prior.parameter in parameter_mapping:
                mapped_name = parameter_mapping[prior.parameter]
                if mapped_name == 'sigma':
                    exported_priors[mapped_name] = {
                        'dist': 'inverse_gamma',
                        'alpha': 1.0,
                        'beta': prior.mean,
                        'source': 'llm_expert',
                        'confidence': prior.confidence,
                        'rationale': prior.rationale
                    }
                else:
                    exported_priors[mapped_name] = {
                        'dist': 'normal',
                        'mu': prior.mean,
                        'sigma': prior.std,
                        'source': 'llm_expert',
                        'confidence': prior.confidence,
                        'rationale': prior.rationale
                    }
        
        logger.info("✓ Priors exported for Bayesian analysis")
        return exported_priors
    
    def save_session_data(self, priors: List[LLMPriorSpecification], filename: Optional[str] = None) -> str:
        """
        セッションデータの保存
        """
        if filename is None:
            filename = f"llm_priors_session_{self.session_id}.json"
        
        session_data = {
            'session_id': self.session_id,
            'model': self.model,
            'temperature': self.temperature,
            'timestamp': datetime.now().isoformat(),
            'priors': [asdict(prior) for prior in priors]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Session data saved to {filename}")
        return filename


class MockLLMPriorElicitor:
    """
    テスト用のモックLLMエリシター
    OpenAI APIキーが利用できない場合のテスト用
    """
    
    def __init__(self, model: str = "mock-gpt-4.1", temperature: float = 0.2):
        """
        初期化
        """
        self.model = model
        self.temperature = temperature
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"✓ Mock LLM Prior Elicitor initialized")
        logger.info(f"  Model: {self.model} (Mock)")
        logger.info(f"  Session ID: {self.session_id}")
    
    def elicit_clinical_priors(self, 
                              treatment_1: str, 
                              treatment_2: str, 
                              outcome_measure: str,
                              clinical_context: str,
                              additional_context: Optional[str] = None) -> List[LLMPriorSpecification]:
        """
        モック版のLLM事前分布設定
        Study2歴史的研究データと実際のtoenailデータセット統計に基づく臨床的合理的な事前分布
        
        データ根拠:
        - 実測ベースライン平均: 1.89mm (Study2では2.5mm想定)
        - 実測月間成長率: 0.555mm/月 (Itraconazole: 0.558, Terbinafine: 0.600)
        - 治療優位性: 平均0.35mm/月 (12ヶ月で0.771mm差)
        - 測定誤差: 全体標準偏差4.39mm
        """
        logger.info("🤖 Using Mock LLM for prior elicitation...")
        logger.info(f"  Context: {clinical_context}")
        logger.info(f"  Comparison: {treatment_2} vs {treatment_1}")
        logger.info(f"  Outcome: {outcome_measure}")
        
        # Study2歴史的データ + 実測データに基づく臨床的に合理的な事前分布
        mock_priors = [
            LLMPriorSpecification(
                parameter="baseline_intercept",
                distribution="normal",
                mean=2.5,  # Study2歴史的想定値（実測1.89mmより保守的）
                std=1.0,   # 適度な不確実性を表現
                confidence=0.80,
                rationale="Historical Study2 baseline assumption (2.5mm) vs actual data mean (1.89mm). Conservative prior allows data to inform.",
                llm_model=self.model,
                timestamp=datetime.now().isoformat(),
                session_id=self.session_id
            ),
            LLMPriorSpecification(
                parameter="time_effect",
                distribution="normal", 
                mean=0.6,   # Study2想定値、実測0.558mmに近い
                std=0.2,    # 合理的な不確実性
                confidence=0.85,
                rationale="Historical Study2 assumption (0.6mm/month) aligns with actual Itraconazole rate (0.558mm/month).",
                llm_model=self.model,
                timestamp=datetime.now().isoformat(),
                session_id=self.session_id
            ),
            LLMPriorSpecification(
                parameter="treatment_advantage",
                distribution="normal",
                mean=0.0,   # Study2中性想定、実測では0.042mm/月差
                std=0.15,   # 小さいが検出可能な効果を許容
                confidence=0.70,
                rationale="Historical Study2 neutral assumption (0.0). Actual data shows 0.042mm/month advantage for Terbinafine.",
                llm_model=self.model,
                timestamp=datetime.now().isoformat(),
                session_id=self.session_id
            ),
            LLMPriorSpecification(
                parameter="error_std",
                distribution="normal",
                mean=3.7,   # Study2歴史的想定（実測4.39mmより保守的）
                std=0.8,    # 測定誤差の不確実性
                confidence=0.85,
                rationale="Historical Study2 error assumption (3.7mm) vs actual data std (4.39mm). Prior informed by clinical experience.",
                llm_model=self.model,
                timestamp=datetime.now().isoformat(),
                session_id=self.session_id
            )
        ]
        
        logger.info(f"✓ Mock LLM generated {len(mock_priors)} prior distributions")
        return mock_priors
    
    def export_priors_for_analysis(self, priors: List[LLMPriorSpecification]) -> Dict:
        """
        ベイズ解析用の事前分布仕様をエクスポート（Mock版）
        """
        exported_priors = {}
        
        parameter_mapping = {
            'baseline_intercept': 'beta_intercept',
            'time_effect': 'beta_time',
            'treatment_advantage': 'beta_interaction',
            'error_std': 'sigma'
        }
        
        for prior in priors:
            if prior.parameter in parameter_mapping:
                mapped_name = parameter_mapping[prior.parameter]
                if mapped_name == 'sigma':
                    exported_priors[mapped_name] = {
                        'dist': 'inverse_gamma',
                        'alpha': 1.0,
                        'beta': prior.mean,
                        'source': 'mock_llm_expert',
                        'confidence': prior.confidence,
                        'rationale': prior.rationale
                    }
                else:
                    exported_priors[mapped_name] = {
                        'dist': 'normal',
                        'mu': prior.mean,
                        'sigma': prior.std,
                        'source': 'mock_llm_expert',
                        'confidence': prior.confidence,
                        'rationale': prior.rationale
                    }
        
        logger.info("✓ Mock priors exported for Bayesian analysis")
        return exported_priors
    
    def save_session_data(self, priors: List[LLMPriorSpecification], filename: Optional[str] = None) -> str:
        """
        セッションデータの保存（Mock版）
        """
        if filename is None:
            filename = f"mock_llm_priors_session_{self.session_id}.json"
        
        session_data = {
            'session_id': self.session_id,
            'model': self.model,
            'temperature': self.temperature,
            'timestamp': datetime.now().isoformat(),
            'priors': [asdict(prior) for prior in priors]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Mock session data saved to {filename}")
        return filename
