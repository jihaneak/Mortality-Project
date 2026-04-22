"""
pathconfig.py — À importer en premier dans chaque page Streamlit
Usage : from pathconfig import PROJECT_ROOT
"""
import os, sys

PROJECT_ROOT = r"C:\Users\PC-HP\Desktop\insea\Mortality Project"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)