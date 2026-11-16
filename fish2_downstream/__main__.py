from pathlib import Path
import sys
import os

sys.dont_write_bytecode = True
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

# Also works through the shared run.py symlink.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fish2_downstream.cli import main

if __name__ == "__main__":
    main()
