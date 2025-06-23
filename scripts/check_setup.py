# filepath: scripts/check_setup.py
"""
Setup Verification Script
セットアップ確認スクリプト
"""

import sys
import os
import subprocess


def check_python_version():
    """Python バージョンの確認"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version OK")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False


def check_rye():
    """Rye の確認"""
    try:
        result = subprocess.run(['rye', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Rye installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Rye not found")
            return False
    except FileNotFoundError:
        print("❌ Rye not installed")
        return False


def check_dependencies():
    """依存関係の確認"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'openai'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing.append(package)
    
    return len(missing) == 0


def check_data_files():
    """データファイルの確認"""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(script_dir, 'data', 'toenail.txt')
    
    if os.path.exists(data_path):
        print("✅ Toenail data file found")
        
        # ファイルサイズ確認
        size = os.path.getsize(data_path)
        print(f"   File size: {size:,} bytes")
        
        return True
    else:
        print("❌ Toenail data file missing")
        return False


def check_modules():
    """ローカルモジュールの確認"""
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(script_dir)
        
        from src import (
            LLMPriorSpecification,
            MockLLMPriorElicitor,
            load_actual_toenail_data,
            comparative_analysis_setup
        )
        print("✅ All modules importable")
        return True
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        return False


def main():
    """メイン確認処理"""
    print("="*50)
    print("🔍 BAYESIAN RCT RESEARCH - SETUP CHECK")
    print("="*50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Rye Package Manager", check_rye), 
        ("Python Dependencies", check_dependencies),
        ("Data Files", check_data_files),
        ("Local Modules", check_modules)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Error during {name} check: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (name, _) in enumerate(checks):
        status = "✅" if results[i] else "❌"
        print(f"{status} {name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! System ready to use.")
        print("Run: python main.py")
    else:
        print(f"\n⚠️  {total - passed} issues found. Please resolve before use.")
        
        if not results[0]:  # Python version
            print("   - Install Python 3.8+")
        if not results[1]:  # Rye
            print("   - Install Rye: curl -sSf https://rye-up.com/get | bash")
        if not results[2]:  # Dependencies  
            print("   - Install dependencies: rye sync")
        if not results[3]:  # Data files
            print("   - Ensure data/toenail.txt exists")
        if not results[4]:  # Modules
            print("   - Check src/ module structure")


if __name__ == "__main__":
    main()
