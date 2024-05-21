#!/usr/bin/env python3.9

import os
import sys

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# Add the project's root directory to the Python path
sys.path.insert(0, project_root)
