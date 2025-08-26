#!/usr/bin/env python3
"""
Setup script for TTC Subway Delay Predictor - Streamlit Version
This script helps set up the environment and verify everything is working.
"""

import subprocess
import sys
import os
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_virtual_environment():
    """Check if running in a virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️  No virtual environment detected (recommended but not required)")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("\n📊 Checking data files...")
    
    required_files = [
        "src/routes/resources/enriched_predictions.csv",
        "src/routes/resources/station_data.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} data file(s). Please ensure all data files are present.")
        return False
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing imports...")
    
    required_packages = [
        "streamlit",
        "pandas", 
        "plotly",
        "numpy"
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - FAILED")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import {len(failed_imports)} package(s): {', '.join(failed_imports)}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚇 TTC Subway Delay Predictor - Streamlit Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    check_virtual_environment()
    
    # Install requirements
    if not install_requirements():
        print("\n💡 Try running: pip install -r requirements_streamlit.txt manually")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n💡 Try reinstalling the requirements or check your Python environment")
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        print("\n💡 Please ensure all data files are present before running the application")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n🚀 To run the application:")
    print("   streamlit run app.py")
    print("\n📖 For more information, see README_STREAMLIT.md")

if __name__ == "__main__":
    main()
