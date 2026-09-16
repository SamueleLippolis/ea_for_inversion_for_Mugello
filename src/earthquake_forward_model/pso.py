"""Global-best particle swarm optimization over the forward-model search space."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from .forward_model import IntensityObservation
from .inversion import EvolutionSearch, Genome, Individual, SearchSpace


@dataclass(frozen=True)
class SwarmSettings:
    swarm_size: int
    iterations: int
    inertia_weight: float = 0.7
    cognitive_weight: float = 1.5
    social_weight: float = 1.5
    maximum_velocity_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.swarm_size < 2 or self.iterations < 0:
            raise ValueError("swarm_size must be >= 2 and iterations nonnegative")
        if any(not math.isfinite(value) or value < 0 for value in (
            self.inertia_weight, self.cognitive_weight, self.social_weight
        )):
            raise ValueError("PSO weights must be finite and nonnegative")
        if not math.isfinite(self.maximum_velocity_fraction) or not (
            0 < self.maximum_velocity_fraction <= 1
        ):
            raise ValueError("maximum_velocity_fraction must be in (0, 1]")


@dataclass
class Particle:
    position: list[float]
    velocity: list[float]
    best_position: list[float]
    best: Individual


class ParticleSwarmSearch(EvolutionSearch):
    """Evaluate rounded particle positions with the shared forward-model fitness."""

    def __init__(self, space: SearchSpace,
                 observations: tuple[IntensityObservation, ...],
                 intensity_law: int, seed: int):
        super().__init__(space, observations, intensity_law, seed)

    def _genome(self, position: list[float]) -> Genome:
        return tuple(min(gene.maximum_index, max(0, math.floor(value + 0.5)))
                     for value, gene in zip(position, self.space.genes))

    def run(self, settings: SwarmSettings,
            progress: Callable[[int, float], None] | None = None) -> Individual:
        particles = []
        global_best: Individual | None = None
        global_best_position: list[float] | None = None
        for _ in range(settings.swarm_size):
            position = [self.rng.uniform(0, gene.maximum_index)
                        for gene in self.space.genes]
            velocity = [self.rng.uniform(-1, 1) * settings.maximum_velocity_fraction
                        * gene.maximum_index for gene in self.space.genes]
            evaluation = self.evaluate(self._genome(position))
            particles.append(Particle(position, velocity, position.copy(), evaluation))
            if global_best is None or evaluation.residual < global_best.residual:
                global_best = evaluation
                global_best_position = position.copy()

        assert global_best is not None and global_best_position is not None
        if progress:
            progress(0, global_best.residual)

        for iteration in range(1, settings.iterations + 1):
            next_positions = []
            for particle in particles:
                new_position = []
                new_velocity = []
                for i, gene in enumerate(self.space.genes):
                    maximum = gene.maximum_index
                    limit = settings.maximum_velocity_fraction * maximum
                    velocity = (settings.inertia_weight * particle.velocity[i]
                                + settings.cognitive_weight * self.rng.random()
                                * (particle.best_position[i] - particle.position[i])
                                + settings.social_weight * self.rng.random()
                                * (global_best_position[i] - particle.position[i]))
                    velocity = min(limit, max(-limit, velocity))
                    position = min(maximum, max(0.0, particle.position[i] + velocity))
                    # A particle that hits a bound should not keep pushing
                    # against it on the next iteration.
                    if position == 0.0 or position == maximum:
                        velocity = 0.0
                    new_position.append(position)
                    new_velocity.append(velocity)
                next_positions.append((new_position, new_velocity))

            for particle, (position, velocity) in zip(particles, next_positions):
                particle.position = position
                particle.velocity = velocity
                evaluation = self.evaluate(self._genome(position))
                if evaluation.residual < particle.best.residual:
                    particle.best = evaluation
                    particle.best_position = position.copy()
                if evaluation.residual < global_best.residual:
                    global_best = evaluation
                    global_best_position = position.copy()
            if progress:
                progress(iteration, global_best.residual)

        return global_best
