#!/usr/bin/env python
"""Simple test runner to generate layout comparison results."""

import sys
import traceback
from pathlib import Path

# Add tests to path
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

try:
    from test_layout_comparison import test_solver_vs_guillotine_and_log
    
    print("Running test_solver_vs_guillotine_and_log...")
    test_solver_vs_guillotine_and_log()
    print("✅ Test passed!")
    
    # Display results
    log_path = Path("Outputs") / "layout_comparison.log"
    if log_path.exists():
        print("\n" + "="*60)
        print("Test Results:")
        print("="*60)
        with open(log_path, 'r') as f:
            print(f.read())
    else:
        print("⚠️  Log file not found at", log_path)
        
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    traceback.print_exc()
    sys.exit(1)
