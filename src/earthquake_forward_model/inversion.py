"""Integer coded GA followed by a distance niching GA for source inversion.

The evolutionary steps follow ``pgakfdeme9.f``. Search bounds and gene steps
are explicit so this can use the independent source parameters of the Python
forward model rather than the Fortran program's event-specific moment/length
coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
import random
from typing import Callable

from .forward_model import IntensityObservation, SourceModelParameters, evaluate_forward_model


PARAMETERS = (
    "latitude", "longitude", "strike", "rake", "dip", "depth",
    "length_plus", "length_minus", "mach_plus", "mach_minus",
    "s_velocity", "seismic_moment",
)
Genome = tuple[int, ...]


@dataclass(frozen=True)
class Gene:
    name: str
    lower: float
    upper: float
    step: float

    def __post_init__(self) -> None:
        lo, hi, step = map(lambda x: Decimal(str(x)),
                           (self.lower, self.upper, self.step))
        if step <= 0 or hi < lo or (hi - lo) % step != 0:
            raise ValueError(f"{self.name}: bounds must contain whole positive steps")

    @property
    def maximum_index(self) -> int:
        return int((Decimal(str(self.upper)) - Decimal(str(self.lower))) /
                   Decimal(str(self.step)))

    def decode(self, index: int) -> float:
        if not 0 <= index <= self.maximum_index:
            raise ValueError(f"{self.name}: gene index out of bounds: {index}")
        return float(Decimal(str(self.lower)) + index * Decimal(str(self.step)))


@dataclass(frozen=True)
class SearchSpace:
    genes: tuple[Gene, ...]
    fixed: dict

    def __post_init__(self) -> None:
        names = tuple(gene.name for gene in self.genes)
        if names != PARAMETERS:
            raise ValueError(f"genes must be ordered as {PARAMETERS}")
        if "samples" not in self.fixed or "sampling_interval" not in self.fixed:
            raise ValueError("fixed values must include samples and sampling_interval")

    def decode(self, genome: Genome) -> SourceModelParameters:
        if len(genome) != len(self.genes):
            raise ValueError("genome length does not match search space")
        values = {gene.name: gene.decode(index)
                  for gene, index in zip(self.genes, genome)}
        return SourceModelParameters(**values, **self.fixed)

    def random_genome(self, rng: random.Random) -> Genome:
        return tuple(rng.randint(0, gene.maximum_index) for gene in self.genes)

    def distance(self, left: Genome, right: Genome) -> float:
        """Mean absolute gene distance normalized by each explored range."""
        terms = [abs(a - b) / gene.maximum_index
                 for a, b, gene in zip(left, right, self.genes)
                 if gene.maximum_index]
        return sum(terms) / len(self.genes)


@dataclass(frozen=True)
class EvolutionSettings:
    population_size: int
    generations: int
    replacement_count: int
    crossover_probability: float = 0.9
    mutation_probability: float = 0.06
    restart_after: int = 10

    def __post_init__(self) -> None:
        if self.population_size < 4 or self.generations < 0:
            raise ValueError("population_size must be >= 4 and generations >= 0")
        if not 2 <= self.replacement_count <= self.population_size - 1:
            raise ValueError("replacement_count must be between 2 and population_size - 1")
        if not 0 <= self.crossover_probability <= 1 or not 0 <= self.mutation_probability <= 1:
            raise ValueError("crossover and mutation probabilities must be in [0, 1]")
        if self.restart_after < 0:
            raise ValueError("restart_after must be nonnegative")


@dataclass(frozen=True)
class Individual:
    genome: Genome
    residual: float


class EvolutionSearch:
    def __init__(self, space: SearchSpace,
                 observations: tuple[IntensityObservation, ...],
                 intensity_law: int, seed: int):
        self.space = space
        self.observations = observations
        self.intensity_law = intensity_law
        self.rng = random.Random(seed)
        self.cache: dict[Genome, float] = {}

    def evaluate(self, genome: Genome) -> Individual:
        if genome not in self.cache:
            model = self.space.decode(genome)
            try:
                residual = evaluate_forward_model(
                    model, self.observations, intensity_law=self.intensity_law).residual
            except ValueError:
                residual = math.inf
            self.cache[genome] = residual
        return Individual(genome, self.cache[genome])

    def _unique(self, genome: Genome, used: set[Genome],
                other_demes: tuple[tuple[Individual, ...], ...],
                radius: float) -> bool:
        if genome in used:
            return False
        return all(self.space.distance(genome, individual.genome) >= radius
                   for deme in other_demes for individual in deme)

    def _repair(self, genome: Genome, used: set[Genome],
                other_demes: tuple[tuple[Individual, ...], ...],
                radius: float) -> Genome:
        if self._unique(genome, used, other_demes, radius):
            return genome
        # The Fortran code resets a randomly chosen allele when it encounters
        # duplicates or a competing deme. Bound attempts make an impossible
        # separation fail explicitly instead of looping forever.
        for _ in range(100):
            i = self.rng.randrange(len(genome))
            changed = list(genome)
            changed[i] = self.rng.randint(0, self.space.genes[i].maximum_index)
            genome = tuple(changed)
            if self._unique(genome, used, other_demes, radius):
                return genome
        for _ in range(1000):
            genome = self.space.random_genome(self.rng)
            if self._unique(genome, used, other_demes, radius):
                return genome
        raise ValueError("cannot fill distinct demes at this distance; reduce the radius")

    def _population(self, size: int, other_demes: tuple[tuple[Individual, ...], ...],
                    radius: float) -> tuple[Individual, ...]:
        used: set[Genome] = set()
        genomes = []
        for _ in range(size):
            genome = self._repair(self.space.random_genome(self.rng),
                                  used, other_demes, radius)
            used.add(genome)
            genomes.append(genome)
        return self._rank(genomes)

    def _rank(self, genomes: list[Genome]) -> tuple[Individual, ...]:
        return tuple(sorted((self.evaluate(genome) for genome in genomes),
                            key=lambda individual: (individual.residual, individual.genome)))

    def _parent(self, population: tuple[Individual, ...]) -> Genome:
        # PGAPack's default selection is a tournament between two candidates.
        first, second = self.rng.sample(population, 2)
        return min((first, second), key=lambda item: item.residual).genome

    def _mutate(self, genome: Genome, probability: float) -> Genome:
        if self.rng.random() >= probability:
            return genome
        i = self.rng.randrange(len(genome))
        changed = list(genome)
        changed[i] = self.rng.randint(0, self.space.genes[i].maximum_index)
        return tuple(changed)

    def _restart(self, population: tuple[Individual, ...],
                 other_demes: tuple[tuple[Individual, ...], ...],
                 radius: float) -> tuple[Individual, ...]:
        best = self._repair(population[0].genome, set(), other_demes, radius)
        genomes = [best]
        used = {best}
        for _ in population[1:]:
            changed = list(best)
            for i, gene in enumerate(self.space.genes):
                if self.rng.random() < 0.5 and gene.maximum_index:
                    changed[i] = min(gene.maximum_index, max(0,
                                     changed[i] + self.rng.choice((-1, 1))))
            genome = self._repair(tuple(changed), used, other_demes, radius)
            used.add(genome)
            genomes.append(genome)
        return self._rank(genomes)

    def _next_generation(self, population: tuple[Individual, ...],
                         settings: EvolutionSettings,
                         other_demes: tuple[tuple[Individual, ...], ...],
                         radius: float) -> tuple[Individual, ...]:
        keep = settings.population_size - settings.replacement_count
        genomes: list[Genome] = []
        used: set[Genome] = set()
        for individual in population[:keep]:
            genome = self._repair(individual.genome, used, other_demes, radius)
            used.add(genome)
            genomes.append(genome)
        while len(genomes) < settings.population_size:
            first, second = self._parent(population), self._parent(population)
            if self.rng.random() < settings.crossover_probability:
                left, right = sorted(self.rng.sample(range(1, len(first)), 2))
                children = (first[:left] + second[left:right] + first[right:],
                            second[:left] + first[left:right] + second[right:])
            else:
                children = (first, second)
            for child in children:
                if len(genomes) == settings.population_size:
                    break
                child = self._mutate(child, settings.mutation_probability)
                child = self._repair(child, used, other_demes, radius)
                used.add(child)
                genomes.append(child)
        return self._rank(genomes)

    def run(self, settings: EvolutionSettings, demes: int = 1,
            radius: float = 0.0,
            progress: Callable[[int, int, float], None] | None = None
            ) -> tuple[tuple[Individual, ...], ...]:
        if demes < 1 or radius < 0 or radius > 1:
            raise ValueError("demes must be positive and radius must be in [0, 1]")
        populations: list[tuple[Individual, ...]] = []
        best_seen: list[float] = []
        stagnant: list[int] = []
        for deme in range(demes):
            population = self._population(settings.population_size,
                                          tuple(populations), radius)
            populations.append(population)
            best_seen.append(population[0].residual)
            stagnant.append(0)
            if progress:
                progress(0, deme + 1, population[0].residual)
        for generation in range(1, settings.generations + 1):
            for deme in range(demes):
                population = self._next_generation(
                    populations[deme], settings, tuple(populations[:deme]), radius)
                if population[0].residual < best_seen[deme]:
                    best_seen[deme] = population[0].residual
                    stagnant[deme] = 0
                else:
                    stagnant[deme] += 1
                if settings.restart_after and stagnant[deme] >= settings.restart_after:
                    population = self._restart(population, tuple(populations[:deme]), radius)
                    stagnant[deme] = 0
                populations[deme] = population
                if progress:
                    progress(generation, deme + 1, population[0].residual)
        return tuple(populations)
