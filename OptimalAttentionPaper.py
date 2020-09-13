import att_paper_utils as utils
import argparse
import numpy as np
import pandas as pd
from concorde.tsp import TSPSolver
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_points', type=int, default=20)
parser.add_argument('--dataset_path', type=str, default='data/attention-paper/tsp20_test_seed1234.pkl')


class GenerateOptimalTSP():
    """
    Generate Concorde Solutions for a TSP dataset
    """

    def __init__(self, data_path, n_points, solve=True):
        self.data = utils.make_dataset(filename=data_path)
        self.data_size = len(self.data)
        self.n_points = n_points
        self.solve = solve
        self.generate_data()

    def generate_data(self):

        points_list = []
        solutions = []
        opt_dists = []

        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Generating data points %i/%i'
                                      % (i+1, self.data_size))

            points = np.array(self.data[i])

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
                sol = solver.solve(time_bound=-1, verbose=False)
                opt_tour, opt_dist = sol.tour, sol.optimal_value/10000
                solutions.append(opt_tour)
                opt_dists.append(opt_dist)

        else:
            solutions = None
            opt_dists = None

        if self.solve:

            print('  [*] Avg Optimal Tour {:.5f} +- {:.5f}'.format(np.mean(opt_dists), 2 * np.std(opt_dists) / np.sqrt(len(opt_dists))))

        data = {'Points': points_list,
                'OptTour': solutions,
                'OptDistance': opt_dists}
        df = pd.DataFrame(data)
        df.to_json(path_or_buf='data/att-TSP'+str(self.n_points)+'-data-test'+'.json')


args = parser.parse_args()

# if __name__ == '__main__':
GenerateOptimalTSP(args.dataset_path, args.n_points)
