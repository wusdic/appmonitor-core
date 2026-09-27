import os, sys

# Single-threaded BLAS / OpenMP (integration notes R15.1 / R19.3): the lib-3
# fits are many tiny eigendecompositions and C-steps, which multithreaded
# OpenBLAS makes 3-10x slower under load. Set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
