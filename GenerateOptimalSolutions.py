import numpy as np
from tqdm import tqdm
import pandas as pd
from concorde.tsp import TSPSolver


class GenerateOptimalTSP():
    """
    Generate TSP dataset
    """

    def __init__(self, data_size, n_points, solve=True,
                 seed=12345):
        self.data_size = data_size
        self.n_points = n_points
        self.solve = solve
        self.seed = seed
        self.data = self.generate_data()
        np.random.seed(seed=seed)

    def generate_data(self):

        points_list = []
        solutions = []
        opt_dists = []

        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Generating data points %i/%i'
                                      % (i+1, self.data_size))

            points = np.random.random((self.n_points, 2))

            points_list.append(points)

        # solutions_iter: for tqdm
        solutions_iter = tqdm(points_list, unit='solve')
        if self.solve:
            for i, points in enumerate(solutions_iter):
                solutions_iter.set_description('Solved %i/%i'
                                               % (i+1, len(points_list)))

                points_scaled = points*10000
                solver = TSPSolver.from_data(points_scaled[:, 0],
                                             points_scaled[:, 1],
                                             'EUC_2D')

                sol = solver.solve(time_bound=-1, verbose=True)

                opt_tour, opt_dist = sol.tour, sol.optimal_value/10000
                solutions.append(opt_tour)
                opt_dists.append(opt_dist)

        else:
            solutions = None
            opt_dists = None

        data = {'Points': points_list,
                'OptTour': solutions,
                'OptDistance': opt_dists}
        df = pd.DataFrame(data)
        df.to_json(path_or_buf='data/data_test'+str(self.n_points)+'.json')


GenerateOptimalTSP(10000, 20)
GenerateOptimalTSP(10000, 50)
GenerateOptimalTSP(10000, 100)
