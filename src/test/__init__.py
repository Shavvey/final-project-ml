import os
import sys

"""
Small testing init script that will provide us with an environment similar to main.py.
Crucial to run this before running the test scripts.
"""
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(PROJECT_PATH, "src")
sys.path.append(SOURCE_PATH)
