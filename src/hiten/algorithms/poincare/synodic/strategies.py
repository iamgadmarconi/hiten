from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase


class _NoOpStrategy(_SeedingStrategyBase):
    def generate(self, *, h0, H_blocks, clmo_table, solve_missing_coord_fn, find_turning_fn):
        raise NotImplementedError("Synodic engine does not generate seeds")
