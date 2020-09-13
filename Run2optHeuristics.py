import argparse
import os
from DataGenerator import TSPDataset
from tqdm import tqdm
import pandas as pd
import mp_utils
import multiprocessing as mp

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


manager = mp.Manager()
return_dict = manager.dict()

c = 0
for i in range(0, len(test_data), args.batch_size):
    batch = test_data[i:i+args.batch_size]
    pbar = tqdm(total=len(batch))
    procs = []
    for j in range(len(batch)):
        if args.heuristic in ['best_improvement', 'first_improvement']:
            proc = mp.Process(target=f,
                              args=(test_data.data_set[c], c, return_dict))
        else:
            proc = mp.Process(target=f,
                              args=(test_data.data_set[c],
                                    args.n_steps,
                                    c, return_dict))
        procs.append(proc)
        proc.start()
        pbar.update(1)
        c += 1

# complete the processes
        for proc in procs:
            proc.join()

# print (return_dict.values())

if args.gat_data_path != '':
    pass
else:
    df = pd.DataFrame.from_dict(return_dict,
                                orient='index',
                                columns=["Tour", "Cost"])

    df.to_csv(args.data_dir+'/heuristics/'
              + args.heuristic
              + '-TSP{}-test-steps{}-testsize{}-batchsize{}.csv'.format(args.n_points,
                                                                   args.n_steps,
                                                                   args.test_size,
                                                                   args.batch_size),
                                                                   index=False)
