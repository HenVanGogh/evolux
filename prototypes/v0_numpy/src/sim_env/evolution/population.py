"""Population manager: species, reproduction, generational turnover."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from sim_env.brain.dynamic_nn import DynamicNN
from sim_env.brain.memory import MemoryBank
from sim_env.creatures.body import Body
from sim_env.creatures.creature import Creature
from sim_env.creatures.genome import Genome, NodeType
from sim_env.creatures.sensors import SensorArray
from sim_env.evolution.directed import DirectedEvolution
from sim_env.evolution.operators import MutationOperator
from sim_env.evolution.selection import SelectionStrategy

if TYPE_CHECKING:
    from sim_env.core.world import World

logger = logging.getLogger(__name__)


class Species:
    """Group of creatures sharing a common ancestor (NEAT-style speciation)."""

    _id_counter = 0

    def __init__(self, representative: Genome) -> None:
        Species._id_counter += 1
        self.species_id = Species._id_counter
        self.representative = representative
        self.members: list[Creature] = []
        self.stagnant_gens: int = 0
        self.best_fitness: float = 0.0

    def add(self, creature: Creature) -> None:
        self.members.append(creature)


class Population:
    """Manages a generation of creatures and drives the evolutionary cycle.

    Responsibilities
    ----------------
    - Spawn initial population from random genomes.
    - Place creatures into the world at the start of each generation.
    - Step all alive creatures each simulation tick.
    - After fitness evaluation: select, cross-over, mutate, produce next gen.
    - Maintain NEAT-style species (optional).
    """

    def __init__(self, cfg: dict, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

        self._pop_size: int = cfg["population"]["size"]
        self._speciation_enabled: bool = cfg["evolution"]["speciation"]["enabled"]
        self._compat_thresh: float = cfg["evolution"]["speciation"]["compatibility_threshold"]
        self._disjoint_c: float = cfg["evolution"]["speciation"]["disjoint_coeff"]
        self._weight_c: float = cfg["evolution"]["speciation"]["weight_coeff"]

        self.selector = SelectionStrategy(cfg)
        self.mutator = MutationOperator(cfg)
        self.directed = DirectedEvolution(cfg)

        self.creatures: list[Creature] = []
        self.species: list[Species] = []

        # Determine sensor size from config
        sense_r = int(cfg["brain"].get("min_hidden", 2))  # reuse as default sense radius
        # Actual sense radius from body default
        self._default_sense_radius = 2
        side = 2 * self._default_sense_radius + 1
        self._n_sensor_inputs = SensorArray.N_SCALAR + side * side
        self._n_action_outputs = 4  # forward, turn_left, turn_right, stay

        self._initialise()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _initialise(self) -> None:
        """Spawn the initial random population."""
        for _ in range(self._pop_size):
            genome = Genome.minimal(
                n_inputs=self._n_sensor_inputs,
                n_outputs=self._n_action_outputs,
                rng=self.rng,
                weight_range=tuple(self.cfg["genome"].get("weight_range", [-1.0, 1.0])),
            )
            creature = self._creature_from_genome(genome)
            self.creatures.append(creature)

        if self._speciation_enabled:
            self._speciate()

        logger.info("Population initialised: %d creatures.", len(self.creatures))

    def _creature_from_genome(self, genome: Genome, x: int = 0, y: int = 0) -> Creature:
        memory = MemoryBank(self.cfg)
        brain = DynamicNN(
            genome=genome,
            n_sensor_inputs=self._n_sensor_inputs,
            n_action_outputs=self._n_action_outputs,
            memory_bank=memory,
        )
        body = Body.decode_from_genome(genome.body_params, self.cfg, x=x, y=y)
        sensors = SensorArray(sense_radius=body.sense_radius)
        return Creature(genome=genome, brain=brain, body=body, sensors=sensors)

    # ── Placement ────────────────────────────────────────────────────────────

    def place_into(self, world: "World") -> None:
        """Scatter creatures at random empty positions in the world."""
        all_positions = [
            (x, y) for y in range(world.height) for x in range(world.width)
        ]
        self.rng.shuffle(all_positions)
        positions = all_positions[: len(self.creatures)]

        for creature, (x, y) in zip(self.creatures, positions):
            creature.body.x, creature.body.y = x, y
            creature.alive = True
            creature.reset_generation()
            world.occupancy[y][x] = creature

    # ── Step ─────────────────────────────────────────────────────────────────

    def step_all(self, world: "World") -> None:
        for creature in self.creatures:
            creature.step(world, self.cfg)

    # ── Evolution ────────────────────────────────────────────────────────────

    def evolve(self, rng: np.random.Generator) -> None:
        """Produce the next generation via selection, crossover, mutation."""
        # Optional directed evolution augmentation
        self.directed.augment_fitness(self.creatures)

        elites = self.selector.get_elites(self.creatures)
        n_offspring = self._pop_size - len(elites)

        parents = self.selector.select_parents(self.creatures, n_offspring * 2, rng)
        offspring: list[Creature] = []

        crossover_rate = self.cfg["evolution"]["operators"].get("crossover_rate", 0.75)
        for i in range(0, len(parents) - 1, 2):
            pa, pb = parents[i], parents[i + 1]
            if rng.random() < crossover_rate:
                # Ensure pa is the fitter parent
                if pb.fitness > pa.fitness:
                    pa, pb = pb, pa
                child_genome = self.mutator.crossover(pa.genome, pb.genome, rng)
            else:
                child_genome = pa.genome.clone()

            child_genome = self.mutator.mutate(child_genome, rng)
            offspring.append(self._creature_from_genome(child_genome))

        # Carry elites forward (re-wrap in fresh body/brain for new generation)
        elite_creatures = []
        for e in elites:
            ec = self._creature_from_genome(e.genome.clone())
            elite_creatures.append(ec)

        self.creatures = elite_creatures + offspring[: self._pop_size - len(elite_creatures)]

        if self._speciation_enabled:
            self._speciate()

    # ── Speciation ────────────────────────────────────────────────────────────

    def _speciate(self) -> None:
        """Assign each creature to a species based on genome compatibility."""
        for s in self.species:
            s.members.clear()

        for creature in self.creatures:
            placed = False
            for s in self.species:
                dist = creature.genome.compatibility_distance(
                    s.representative,
                    self._disjoint_c,
                    self._weight_c,
                )
                if dist < self._compat_thresh:
                    s.add(creature)
                    placed = True
                    break
            if not placed:
                new_species = Species(creature.genome.clone())
                new_species.add(creature)
                self.species.append(new_species)

        # Remove empty species
        self.species = [s for s in self.species if s.members]

        # Update representatives
        for s in self.species:
            s.representative = self.rng.choice(s.members).genome.clone()  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return (
            f"Population(size={len(self.creatures)}, species={len(self.species)})"
        )
