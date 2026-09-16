from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake_forward_model.inversion import Gene, Individual, PARAMETERS, SearchSpace
from earthquake_forward_model.pso import ParticleSwarmSearch, SwarmSettings


class QuadraticSwarm(ParticleSwarmSearch):
    def evaluate(self, genome):
        assert all(0 <= value <= 10 for value in genome)
        return Individual(genome, float(sum(value * value for value in genome)))


class ParticleSwarmTest(unittest.TestCase):
    def setUp(self):
        self.space = SearchSpace(
            tuple(Gene(name, 0, 10, 1) for name in PARAMETERS),
            {"samples": 512, "sampling_interval": 0.05})

    def test_repeatable_bounded_search_with_nonincreasing_best(self):
        settings = SwarmSettings(12, 8)

        def run():
            history = []
            search = QuadraticSwarm(self.space, (), 2, 333)
            best = search.run(settings, progress=lambda iteration, residual:
                              history.append((iteration, residual)))
            return best, history

        best, history = run()
        self.assertEqual((best, history), run())
        self.assertEqual(len(history), 9)
        self.assertEqual([item[0] for item in history], list(range(9)))
        self.assertEqual([item[1] for item in history],
                         sorted((item[1] for item in history), reverse=True))
        self.assertEqual(best.residual, history[-1][1])

    def test_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            SwarmSettings(1, 10)
        with self.assertRaises(ValueError):
            SwarmSettings(10, -1)
        with self.assertRaises(ValueError):
            SwarmSettings(10, 10, maximum_velocity_fraction=0)


if __name__ == "__main__":
    unittest.main()
