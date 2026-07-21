"""Tree-model placeholders."""

from __future__ import annotations


class TreeBenchmark:
    """TODO wrapper for random forest and gradient-boosted tree benchmarks."""

    def fit(self, *_: object, **__: object) -> "TreeBenchmark":
        raise NotImplementedError("Tree benchmarks are planned but not implemented yet")

