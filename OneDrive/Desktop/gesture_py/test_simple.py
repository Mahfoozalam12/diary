#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from ytgesture import GestureRecognizer
    print('Creating GestureRecognizer...')
    gr = GestureRecognizer()
    print('Success!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()