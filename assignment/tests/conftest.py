import sys
from pathlib import Path

# Ensure workspace root and assignment root are in sys.path
TEST_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = TEST_DIR.parent
WORKSPACE_DIR = ASSIGNMENT_DIR.parent

for p in [str(WORKSPACE_DIR), str(ASSIGNMENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
