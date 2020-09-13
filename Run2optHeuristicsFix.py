import argparse
import os
import istarmap
from DataGenerator import TSPDataset
from tqdm import tqdm
import pandas as pd
import new_mp_utils as mp_utils
import multiprocessing as mp
import time

parser = argparse.ArgumentParser()


parser.add_argument('--heuristic',
                    default=None, type=str, help='Heuristic to run')
parser.add_argument('--test_size',
                    default=512, type=int, help='Test data size')
parser.add_argument('--batch_size',
                    default=512, type=int, help='batch size')
parser.add_argument('--n_points',
                    type=int, default=100, help='Number of points in TSP')
parser.add_argument('--n_steps',
                    default=1000,
                    type=int, help='Number of steps to run')
parser.add_argument('--test_from_data',
                    default=True,
                    action='store_true', help='read TSP data')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--gat_data_path', type=str, default='')

args = parser.parse_args()


# Heuristics handling
heuristics_list = ['best_improvement',
                   'first_improvement',
                   'best_improvement_restart',
                   'first_improvement_restart']

if args.heuristic is None:
    raise Exception('Heuristics cannot be empty.')
elif args.heuristic not in heuristics_list:
    raise Exception('Heuristic not implemented.')

if args.heuristic == 'best_improvement':
    args.n_steps = 0
    f = mp_utils.heuristic_2opt_bi
elif args.heuristic == 'first_improvement':
    args.n_steps = 0
    f = mp_utils.heuristic_2opt_fi
elif args.heuristic == 'best_improvement_restart':
    f = mp_utils.heuristic_2opt_bi_restart
elif args.heuristic == 'first_improvement_restart':
    f = mp_utils.heuristic_2opt_fi_restart

# Data handling
if args.test_from_data:
    if args.gat_data_path != '':
        # test_data
        pass
    else:
        test_data = TSPDataset(dataset_fname=os.path.join(args.data_dir,
                                                          'TSP{}-data-test.json'
                                                          .format(args.n_points)),
                               num_samples=args.test_size, seed=1234)

else:
    raise Exception('Heuristic will only run for saved data.')


num_cpus = os.cpu_count()


result = []
start = time.perf_counter()
for i in range(0, len(test_data), args.batch_size):
    batch = test_data[i:i+args.batch_size]
    steps = [args.n_steps for _ in range(len(batch))]
    with mp.Pool(num_cpus) as p:
        if args.heuristic in ['best_improvement', 'first_improvement']:
            r = list(tqdm(p.imap(f, batch), total=len(batch)))
        else:

            r = list(tqdm(p.istarmap(f, zip(batch, steps)), total=len(batch)))
    result.extend(r)
finish = time.perf_counter()
print("Finished in {:.2f} secs".format(finish-start))

df = pd.DataFrame(result, columns=["Cost"])
print(df)

df.to_csv(args.data_dir+'/heuristics/'
          + args.heuristic
          + 'aaa-TSP{}-test-steps{}-testsize{}-batchsize{}.csv'.format(args.n_points,
                                                               args.n_steps,
                                                               args.test_size,
                                                               args.batch_size),
                                                               index=False)
