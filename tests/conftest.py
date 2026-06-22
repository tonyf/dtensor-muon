import os
import sys

import torch

# Make shared test helpers importable from every test module, regardless of which
# subdirectory it lives in. The insertions are also propagated to subprocesses
# launched by testkit.run_distributed (multiprocessing forwards sys.path).
TESTS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TESTS_DIR)

# The orthogonalization routines are decorated with @torch.compile(fullgraph=True).
# The correctness tests sweep many (shape, steps) combinations in a single pytest
# process, which otherwise blows past Dynamo's default recompile cache limit and
# hard-fails with FailOnRecompileLimitHit (fullgraph=True turns the limit into an
# error rather than an eager fallback). Raise the limits so every parametrization
# can compile within one process.
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 256
