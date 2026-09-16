import random
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake_forward_model.inversion import (
    EvolutionSearch, EvolutionSettings, Gene, Individual, PARAMETERS, SearchSpace,
)


class QuadraticSearch(EvolutionSearch):
    def evaluate(self, genome):
        return Individual(genome, float(sum(value * value for value in genome)))


class FlatSearch(EvolutionSearch):
    def __init__(self, *args):
        super().__init__(*args)
        self.restarts = 0

    def evaluate(self, genome):
        return Individual(genome, 1.0)

    def _restart(self, population, other_demes, radius):
        self.restarts += 1
        return super()._restart(population, other_demes, radius)


class InversionTest(unittest.TestCase):
    def setUp(self):
        self.space = SearchSpace(
            tuple(Gene(name, 0, 10, 1) for name in PARAMETERS),
            {"samples": 512, "sampling_interval": 0.05})
        self.settings = EvolutionSettings(12, 4, 4)

    def test_gene_distance_and_bounds(self):
        self.assertEqual(self.space.distance((0,) * 12, (10,) * 12), 1.0)
        self.assertEqual(self.space.distance((0,) * 12, (0,) * 12), 0.0)
        self.assertTrue(all(0 <= value <= 10
                            for value in self.space.random_genome(random.Random(1))))

    def test_repeatable_elitist_niching_run(self):
        def run():
            history = []
            search = QuadraticSearch(self.space, (), 2, 333)
            demes = search.run(self.settings, demes=2, radius=0.085,
                               progress=lambda generation, deme, best:
                               history.append((generation, deme, best)))
            return demes, history

        demes, history = run()
        self.assertEqual((demes, history), run())
        for deme in (1, 2):
            best = [value for _, index, value in history if index == deme]
            self.assertEqual(best, sorted(best, reverse=True))
        self.assertEqual(len(set(ind.genome for ind in demes[0])), 12)
        self.assertEqual(len(set(ind.genome for ind in demes[1])), 12)
        self.assertTrue(all(self.space.distance(a.genome, b.genome) >= 0.085
                            for a in demes[0] for b in demes[1]))

    def test_restart_after_stagnation(self):
        search = FlatSearch(self.space, (), 2, 333)
        search.run(EvolutionSettings(12, 3, 4, restart_after=2))
        self.assertEqual(search.restarts, 1)


if __name__ == "__main__":
    unittest.main()
