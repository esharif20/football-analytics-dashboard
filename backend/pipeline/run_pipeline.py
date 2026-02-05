#!/usr/bin/env python3
"""
Wrapper to run the football analysis pipeline with correct Python path setup.

This script ensures all relative imports work correctly by:
1. Adding the src directory to Python path
2. Running main() from the correct context
"""
import sys
import os

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')

# Add src to the beginning of Python path
sys.path.insert(0, src_dir)

# Change to src directory so relative paths work
os.chdir(src_dir)

# Now import and run main
if __name__ == "__main__":
    from main import main
    main()
