#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from ytgesture import GestureRecognizer
    print("Creating GestureRecognizer...")
    gr = GestureRecognizer()
    print("GestureRecognizer created successfully!")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()