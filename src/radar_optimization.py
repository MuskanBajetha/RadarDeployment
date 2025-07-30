# src/radar_optimization.py

import numpy as np
import pygad
from radar_simulation import get_radar_coverage

def generate_heuristic_population(dem, invalid_mask, num_radars, population_size=5):
    """
    Generate initial population using heuristic rules (high elevation, flat terrain, spacing).
    """
    valid_indices = np.argwhere(~invalid_mask)
    elevation = dem[~invalid_mask]
    # Pick top 20% high points
    threshold = np.percentile(elevation, 80)
    high_points = valid_indices[dem[~invalid_mask] >= threshold]

    population = []
    for _ in range(population_size):
        solution = []
        chosen = []
        for _ in range(num_radars):
            # Pick from high points but with randomness
            idx = np.random.choice(len(high_points))
            y, x = high_points[idx]
            # Enforce spacing
            if all(np.linalg.norm(np.array([y, x]) - np.array(c)) > 5 for c in chosen):
                chosen.append((y, x))
        # If insufficient spacing caused fewer picks, fill with random valid points
        while len(chosen) < num_radars:
            idx = np.random.choice(len(valid_indices))
            chosen.append(tuple(valid_indices[idx]))
        solution.extend([coord for point in chosen for coord in point])
        population.append(solution)
    return np.array(population)


class RadarOptimizer:
    def __init__(self, dem, invalid_mask, num_radars, radius_pixels, radar_height):
        self.dem = dem
        self.invalid_mask = invalid_mask
        self.num_radars = num_radars
        self.radius_pixels = radius_pixels
        self.radar_height = radar_height
        self.height, self.width = dem.shape

        # Only consider valid terrain
        self.valid_indices = np.argwhere(~invalid_mask)

    def fitness_func(self, ga_instance, solution, solution_idx):
        # solution = [y1, x1, y2, x2, ..., yN, xN]
        solution = np.array(solution, dtype=int).reshape((self.num_radars, 2))

        #ye rha progress dikahne vala code
        gen = ga_instance.generations_completed + 1
        print(f"[GEN {gen}] Evaluating solution {solution_idx + 1}/{ga_instance.sol_per_pop}")


        covered = np.zeros_like(self.dem, dtype=bool)
        for y, x in solution:
            if 0 <= y < self.height and 0 <= x < self.width:
                coverage = get_radar_coverage(self.dem, (x, y), self.radius_pixels, self.radar_height)
                coverage &= ~self.invalid_mask
                covered |= coverage

        fitness = np.sum(covered)
        return fitness

    def optimize(self, generations=4, sol_per_pop=4):
        gene_space = [{'low': 0, 'high': self.height - 1}, {'low': 0, 'high': self.width - 1}] * self.num_radars

        # --- SHI: Generate heuristic initial population ---
        initial_population = generate_heuristic_population(
            self.dem, self.invalid_mask, self.num_radars, population_size=sol_per_pop
        )

        ga_instance = pygad.GA(
            gene_space=[{'low': 0, 'high': self.height - 1}, {'low': 0, 'high': self.width - 1}] * self.num_radars,
            num_generations=generations,
            num_parents_mating=sol_per_pop // 2,
            fitness_func=self.fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=self.num_radars * 2,
            initial_population=initial_population,   # <-- KEY CHANGE
            mutation_type="random",
            mutation_percent_genes=20,
            stop_criteria=["saturate_10"]
        )



        '''
        ga_instance = pygad.GA(
            gene_space=gene_space,
            num_generations=generations,
            num_parents_mating=sol_per_pop // 2,
            fitness_func=self.fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=self.num_radars * 2,
            mutation_type="random",
            mutation_percent_genes=20,
            stop_criteria=["saturate_10"]
        )
        '''
        ga_instance.run()
        ga_instance.plot_fitness(title="📈 PyGAD Optimization Progress")

        best_solution, _, _ = ga_instance.best_solution()
        optimized_positions = np.array(best_solution, dtype=int).reshape((self.num_radars, 2))
        return optimized_positions
