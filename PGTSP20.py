import argparse
import uuid
import os
import torch
import json
import numpy as np
import torch.backends.cudnn as cudnn
from utils import AverageMeter
from torch.optim import Adam, lr_scheduler, RMSprop
from torch.utils.data import DataLoader
from ActorCriticNetwork import ActorCriticNetwork
from DataGenerator import TSPDataset
from TSPEnvironment import TSPInstanceEnv, VecEnv
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()

# ----------------------------------- Data ---------------------------------- #
parser.add_argument('--train_size',
                    default=5120, type=int, help='Training data size')
parser.add_argument('--test_size',
                    default=256, type=int, help='Test data size')
parser.add_argument('--test_from_data',
                    default=True, action='store_true', help='Test data size')
parser.add_argument('--batch_size',
                    default=512, type=int, help='Batch size')
parser.add_argument('--n_points',
                    type=int, default=20, help='Number of points in the TSP')

# ---------------------------------- Train ---------------------------------- #
parser.add_argument('--n_steps',
                    default=200,
                    type=int, help='Number of steps in each episode')
parser.add_argument('--n',
                    default=8,
                    type=int, help='Number of steps to bootstrap')
parser.add_argument('--gamma',
                    default=0.99,
                    type=float, help='Discount factor for rewards')
parser.add_argument('--render',
                    default=False,
                    action='store_true', help='Render')
parser.add_argument('--render_from_epoch',
                    default=0,
                    type=int, help='Epoch to start rendering')
parser.add_argument('--update_value',
                    default=False,
                    action='store_true',
                    help='Use the value function for TD updates')
parser.add_argument('--epochs',
                    default=200, type=int, help='Number of epochs')
parser.add_argument('--lr',
                    type=float, default=0.001, help='Learning rate')
parser.add_argument('--wd',
                    default=1e-5,
                    type=float, help='Weight decay')
parser.add_argument('--beta',
                    type=float, default=0.005, help='Entropy loss weight')
parser.add_argument('--zeta',
                    type=float, default=0.5, help='Value loss weight')
parser.add_argument('--max_grad_norm',
                    type=float, default=0.3, help='Maximum gradient norm')
parser.add_argument('--no_norm_return',
                    default=False,
                    action='store_true', help='Disable normalised returns')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--rms_prop',
                    default=False,
                    action='store_true', help='Disable normalised returns')
parser.add_argument('--adam_beta1',
                    type=float, default=0.9, help='ADAM beta 1')
parser.add_argument('--adam_beta2',
                    type=float, default=0.999, help='ADAM beta 2')
# ----------------------------------- GPU ----------------------------------- #
parser.add_argument('--gpu',
                    default=True, action='store_true', help='Enable gpu')
parser.add_argument('--gpu_n',
                    default=1, type=int, help='Choose GPU')
# --------------------------------- Network --------------------------------- #
parser.add_argument('--input_dim',
                    type=int, default=2, help='Input size')
parser.add_argument('--embedding_dim',
                    type=int, default=128, help='Embedding size')
parser.add_argument('--hidden_dim',
                    type=int, default=128, help='Number of hidden units')
parser.add_argument('--n_rnn_layers',
                    type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--n_actions',
                    type=int, default=2, help='Number of nodes to output')
parser.add_argument('--graph_ref',
                    default=False,
                    action='store_true',
                    help='Use message passing as reference')

# ----------------------------------- Misc ---------------------------------- #
parser.add_argument("--name", type=str, default="", help="Name of the run")
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--model_dir', type=str, default='models')

# unique id in case of no name given
uid = uuid.uuid4()
id = uid.hex

# create {} to log stuff
log = {}
log['hyperparameters'] = {}
args = parser.parse_args()

# log hyperparameters
for arg in vars(args):
    log['hyperparameters'][arg] = getattr(args, arg)

# give it a clever name :D
if args.name != '':
    id = args.name
print("Name:", str(id))

# select a gpu to use
if args.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, {} devices available.'.format(torch.cuda.device_count()))
    torch.cuda.set_device(args.gpu_n)
    print("GPU: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device("cuda")
else:
    USE_CUDA = False
    device = torch.device("cpu")


# if loading the model from file add it here
if args.load_path != '':
    print('  [*] Loading model from {}'.format(args.load_path))

    policy = torch.load(
        os.path.join(os.getcwd(), args.load_path))

else:
    # create actor-critic network
    policy = ActorCriticNetwork(args.input_dim,
                                args.embedding_dim,
                                args.hidden_dim,
                                args.n_points,
                                args.n_rnn_layers,
                                args.n_actions,
                                args.graph_ref)


# define the optimizer and scheduler
if args.rms_prop:
    optimizer = RMSprop(policy.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    optimizer = Adam(policy.parameters(),
                     lr=args.lr,
                     weight_decay=args.wd,
                     betas=(args.adam_beta1, args.adam_beta2))

scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)


# Move policy to the GPU - Use more than one GPU if available
if USE_CUDA:
    policy.cuda()
    # policy = torch.nn.DataParallel(policy,
    #                               device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


if args.test_from_data:
    test_data = TSPDataset(dataset_fname=os.path.join(args.data_dir,
                                                      'TSP{}-data.json'
                                                      .format(args.n_points)),
                           num_samples=args.test_size)
else:
    test_data = TSPDataset(dataset_fname=None,
                           size=args.n_points,
                           num_samples=args.test_size)

# load the test data
test_loader = DataLoader(test_data,
                         batch_size=args.test_size,
                         shuffle=False,
                         num_workers=6)


# buffer to store experiences
class buffer:

    def __init__(self):
        # action & reward buffer
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []

    def clear_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.entropies[:]


def select_action(state, hidden, buffer, best_state):

    probs, action, log_probs_action, v, entropy, hidden = policy(state,
                                                                 best_state,
                                                                 hidden)
    buffer.log_probs.append(log_probs_action)
    buffer.states.append(state)
    buffer.actions.append(action)
    buffer.values.append(v)
    buffer.entropies.append(entropy)
    return action, v, hidden


def learn(R, t_s, beta, zeta, count_learn, epoch):
    """
    Training. Calcultes actor and critic losses and performs backprop.
    """

    count_steps = 0
    sum_returns = 0.0
    sum_advantage = 0.0
    sum_loss_actor = 0.0
    sum_loss_critic = 0.0
    sum_entropy = 0.0
    sum_loss_total = 0.0
    sum_grads_l2 = 0.0
    sum_grads_max = 0.0
    sum_grads_var = 0.0

    # Starting sum of losses for logging
    if t_s == 0:
        epoch_train_policy_loss.reset()
        epoch_train_entropy_loss.reset()
        epoch_train_value_loss.reset()
        epoch_train_loss.reset()

    # Returns
    if R is None:
        R = torch.zeros((args.batch_size, 1)).to(device)
    returns = []  # returns for each state discounted
    for s in reversed(range(len(buffer.rewards))):
        R = buffer.rewards[s] + args.gamma * R
        returns.insert(0, R)

    returns = torch.stack(returns).detach()
    if not args.no_norm_return:
        r_mean = returns.mean()
        r_std = returns.std()
        eps = np.finfo(np.float32).eps.item()  # small number to avoid div/0
        returns = (returns - r_mean)/(r_std + eps)

    # num of experiences in this "batch" of experiences
    n_experiences = args.batch_size*args.n
    # transform lists to tensor
    values = torch.stack(buffer.values)
    log_probs = torch.stack(buffer.log_probs).mean(2).unsqueeze(2)
    entropies = torch.stack(buffer.entropies).mean(2).unsqueeze(2)
    advantages = returns - values
    p_loss = (-log_probs*advantages.detach()).mean()
    v_loss = zeta*(returns - values).pow(2).mean()
    e_loss = (0.9**(epoch+1))*beta*entropies.sum(0).mean()

    optimizer.zero_grad()

    p_loss.backward(retain_graph=True)
    grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                            for p in policy.parameters()
                            if p.grad is not None])

    r_loss = - e_loss + v_loss

    r_loss.backward()
    # nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
    optimizer.step()
    loss = p_loss + r_loss

    # track statistics
    sum_returns += returns.mean()
    sum_advantage += advantages.mean()
    sum_loss_actor += p_loss
    sum_loss_critic += v_loss
    sum_loss_total += loss
    sum_entropy += e_loss

    sum_grads_l2 += np.sqrt(np.mean(np.square(grads)))
    sum_grads_max += np.max(np.abs(grads))
    sum_grads_var += np.var(grads)

    count_steps += 1

    writer.add_scalar("Returns", sum_returns/count_steps, count_learn)
    writer.add_scalar("Advantage", sum_advantage/count_steps, count_learn)
    writer.add_scalar("Loss_Actor", sum_loss_actor/count_steps, count_learn)
    writer.add_scalar("Loss_Critic", sum_loss_critic/count_steps, count_learn)
    writer.add_scalar("Loss_Entropy", sum_entropy/count_steps, count_learn)
    writer.add_scalar("Loss_Total", sum_loss_total/count_steps, count_learn)

    writer.add_scalar("Gradients_L2", sum_grads_l2/count_steps, count_learn)
    writer.add_scalar("Gradients_Max", sum_grads_max/count_steps, count_learn)
    writer.add_scalar("Gradients_Var", sum_grads_var/count_steps, count_learn)

    epoch_train_policy_loss.update(p_loss.item(), n_experiences)
    epoch_train_entropy_loss.update(e_loss.item()/args.n, n_experiences)
    epoch_train_value_loss.update(v_loss.item(), n_experiences)
    epoch_train_loss.update(loss.item(), n_experiences)

    buffer.clear_buffer()


# Initiate the buffer
buffer = buffer()

# Initiate the logs
epoch_train_policy_loss = AverageMeter()
train_policy_loss_log = AverageMeter('train_policy_loss')

epoch_train_entropy_loss = AverageMeter()
train_entropy_loss_log = AverageMeter('train_entropy_loss')

epoch_train_value_loss = AverageMeter()
train_value_loss_log = AverageMeter('train_value_loss')

epoch_train_loss = AverageMeter()
train_loss_log = AverageMeter('train_loss')

train_rwd_log = AverageMeter('train_reward')
train_init_dist_log = AverageMeter('train_init_dist')
train_best_dist_log = AverageMeter('train_best_dist')


val_rwd_log = AverageMeter('val_reward')
val_init_dist_log = AverageMeter('val_init_dist')
val_best_dist_log = AverageMeter('val_best_dist')


best_running_reward = 0
val_best_dist = 1e10
best_gap = 1e10
count_learn = 0

writer = SummaryWriter(comment="-pg_" + args.name)

for epoch in range(args.epochs):
    # training
    train_data = TSPDataset(dataset_fname=None,
                            size=args.n_points,
                            num_samples=args.train_size)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=6)
    # save metrics for all batches
    epoch_rewards = []
    epoch_initial_distances = []
    epoch_best_distances = []

    #TSP 20
    if epoch == 100:
        args.n = 10

    if epoch == 150:
        args.n = 20

    for batch_idx, batch_sample in enumerate(train_loader):
        t = 0
        b_sample = batch_sample.clone().detach().numpy()
        batch_reward = 0

        # every batch defines a set of agents running the same policy
        env = VecEnv(TSPInstanceEnv, b_sample.shape[0], args.n_points)
        state, initial_distance, best_state = env.reset(b_sample)
        hidden = None

        while t < args.n_steps:
            t_s = t
            while t - t_s < args.n and t != args.n_steps:

                if args.render and epoch > args.render_from_epoch:
                    env.render()
                state = torch.from_numpy(state).float().to(device)
                best_state = torch.from_numpy(best_state).float().to(device)
                action, v, _ = select_action(state,
                                             hidden,
                                             buffer,
                                             best_state)

                next_state, reward, _, best_distance, _, next_best_state = \
                    env.step(action.cpu().numpy())

                buffer.rewards.append(torch.from_numpy(reward).float().to(device))
                batch_reward += reward

                state = next_state
                best_state = next_best_state
                t += 1
            if args.update_value:
                next_state = torch.from_numpy(next_state).float().to(device)
                next_best_state = torch.from_numpy(best_state).float().to(device)
                _, _, _, next_v, _, _ = policy(next_state, next_best_state, hidden)
                R = next_v
            else:
                R = None
            count_learn += 1
            learn(R, t_s, args.beta, args.zeta, count_learn, epoch)

        epoch_rewards.append(batch_reward)
        epoch_best_distances.append(best_distance)
        epoch_initial_distances.append(initial_distance)

    epoch_reward = np.mean(epoch_rewards)
    epoch_initial_distance = np.mean(epoch_initial_distances)
    epoch_best_distance = np.mean(epoch_best_distances)

    train_policy_loss_log.update(epoch_train_policy_loss.avg)
    train_entropy_loss_log.update(epoch_train_entropy_loss.avg)
    train_value_loss_log.update(epoch_train_value_loss.avg)
    train_loss_log.update(epoch_train_loss.avg)

    train_rwd_log.update(epoch_reward)
    train_init_dist_log.update(epoch_initial_distance)
    train_best_dist_log.update(epoch_best_distance)

    # validation
    val_epoch_rewards = []
    val_epoch_best_distances = []
    val_epoch_initial_distances = []
    sum_probs = 0
    for val_batch_idx, val_batch_sample in enumerate(test_loader):
        val_b_sample = val_batch_sample.clone().detach().numpy()
        val_batch_reward = 0
        env = VecEnv(TSPInstanceEnv, val_b_sample.shape[0], args.n_points)
        state, initial_distance, best_state = env.reset(val_b_sample)
        t = 0
        hidden = None
        while t < args.n_steps:
            state = torch.from_numpy(state).float().to(device)
            best_state = torch.from_numpy(best_state).float().to(device)
            with torch.no_grad():
                probs, action, _, _, _, _ = policy(state, best_state, hidden)
            sum_probs += probs
            action = action.cpu().numpy()
            state, reward, _, best_distance, distance, best_state = env.step(action)
            val_batch_reward += reward
            t += 1

        val_epoch_rewards.append(val_batch_reward)
        val_epoch_best_distances.append(best_distance)
        val_epoch_initial_distances.append(initial_distance)

    avg_probs = torch.sum(sum_probs, dim=0)/(args.n_steps*args.test_size)*100
    avg_probs = avg_probs.cpu().numpy().round(2)
    val_epoch_reward = np.mean(val_epoch_rewards)
    val_epoch_best_distance = np.mean(val_epoch_best_distances)
    val_epoch_initial_distance = np.mean(val_epoch_initial_distances)

    val_rwd_log.update(val_epoch_reward)
    val_init_dist_log.update(val_epoch_initial_distance)
    val_best_dist_log.update(val_epoch_best_distance)

    scheduler.step()

    writer.add_scalar("Rewards_Training",
                      epoch_reward,
                      epoch)
    writer.add_scalar("Rewards_Testing",
                      val_epoch_reward,
                      epoch)
    writer.add_scalar("Tour_Cost_Training",
                      train_best_dist_log.val/10000,
                      epoch)
    writer.add_scalar("Tour_Cost_Testing",
                      val_best_dist_log.val/10000,
                      epoch)
    if args.test_from_data:
        gap = ((val_best_dist_log.val/10000)/np.mean(test_data.opt) - 1.0)*100
        writer.add_scalar("Gap_Testing",
                          gap,
                          epoch)
    if val_rwd_log.exp_avg > best_running_reward \
       or val_best_dist_log.val < val_best_dist\
       or (args.test_from_data and gap < best_gap):

        print('\033[1;37;40m Saving model...\033[0m')
        model_dir = os.path.join(args.model_dir, str(id))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        checkpoint = {
            'policy': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(model_dir,
                                            'pg-{}-TSP{}-epoch-{}.pt'
                                            .format(str(id), args.n_points,
                                                    epoch)))
        torch.save(policy, os.path.join(model_dir,
                                        'full-model-pg-{}-TSP{}-epoch-{}.pt'
                                        .format(str(id), args.n_points,
                                                epoch)))
        best_running_reward = val_rwd_log.exp_avg
        val_best_dist = val_best_dist_log.val
        best_gap = gap

    if epoch % args.log_interval == 0:

        train_rwd_log.log(log)
        train_init_dist_log.log(log)
        train_best_dist_log.log(log)

        train_policy_loss_log.log(log)
        train_entropy_loss_log.log(log)
        train_value_loss_log.log(log)
        train_loss_log.log(log)

        val_rwd_log.log(log)
        val_init_dist_log.log(log)
        val_best_dist_log.log(log)

        print('\033[1;32;40m Train - epoch:{} |rwd: {:.2f}'
              .format(epoch, train_rwd_log.val),
              '|running rwd: {:.2f} |best cost: {:.3f}\033[0m'
              .format(train_rwd_log.exp_avg, train_best_dist_log.val/10000))

        if not args.test_from_data:

            print('\033[1;33;40m Valid - epoch:{} |rwd: {:.2f}'
                  .format(epoch, val_rwd_log.val),
                  '|running rwd: {:.2f} |best cost: {:.2f}\033[0m'
                  .format(val_rwd_log.exp_avg, val_best_dist_log.val/10000))
        else:
            print('\033[1;33;40m Valid - epoch:{} |rwd: {:.2f}'
                  .format(epoch, val_rwd_log.val),
                  '|running rwd: {:.2f} |best cost: {:.3f}'
                  .format(val_rwd_log.exp_avg, val_best_dist_log.val/10000),
                  '|optimal cost: {:.3f} |gap {:.3f}\033[0m'
                  .format(np.mean(test_data.opt), gap))

        # print("\033[1;37;40m Probabilities: \n",
        #       np.array2string(avg_probs,
        #                       precision=1, separator=' ',
        #                       suppress_small=True), "\033[0m")

        with open(os.path.join(args.log_dir,
                               'pg-{}-TSP{}.json'
                               .format(str(id),
                                       args.n_points)), 'w') as outfile:
            json.dump(log, outfile, indent=4)
