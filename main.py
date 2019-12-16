import maml_rl.envs
import gym
import numpy as np
import torch
import json
import copy
import pickle

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.lstm_tree import TreeLSTM

from teachers.teacher_controller import TeacherController

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0'])

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)

    
    if args.env_name == 'AntVel-v1':
        param_bounds = {"goal": [0, 3]}

    if args.env_name == 'AntPos-v0':
        param_bounds = {"x": [-3, 3],
                        "y": [-3, 3]}

    teacher = TeacherController(args.teacher, args.nb_test_episodes, param_bounds, seed=args.seed, teacher_params={})
    tree = TreeLSTM(args.tree_hidden_layer, len(param_bounds.keys()), args.cluster_0, args.cluster_1, device=args.device)
    tree.to(args.device)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape) + args.tree_hidden_layer),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers, tree=tree)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape) + args.tree_hidden_layer),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers, tree=tree)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))



    metalearner = MetaLearner(sampler, policy, baseline, tree, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)
    
    all_tasks = []
    # torch.autograd.set_detect_anomaly(True)
    for batch in range(args.num_batches):
        print("starting iteration {}".format(batch))
        tasks = []
        for _ in range(args.meta_batch_size):
            if args.env_name == 'AntPos-v0':
                tasks.append({"position": teacher.task_generator.sample_task()})
            if args.env_name == 'AntVel-v1':
                tasks.append({"velocity": teacher.task_generator.sample_task()[0]})
        all_tasks.append(tasks)
       # tasks = np.array(tasks)
        # tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        with open('./logs/{0}/task_list.pkl'.format(args.output_folder), 'wb') as pf:
            pickle.dump(all_tasks, pf)

        print("collecting data...".format(batch))
        episodes = metalearner.sample(tasks, first_order=args.first_order)
        print("training...".format(batch))
        metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)

        # Tensorboard
        writer.add_scalar('total_rewards/before_update',
            total_rewards([ep.rewards for ep, _ in episodes]), batch)
        writer.add_scalar('total_rewards/after_update',
            total_rewards([ep.rewards for _, ep in episodes]), batch)
        
        tr = [ep.rewards for _, ep in episodes]
        tr = [torch.mean(torch.sum(rewards, dim=0)).item() for rewards in tr]
        print("rewards:", tr)
        for t in range(args.meta_batch_size):
            if args.env_name == 'AntPos-v0':
                teacher.task_generator.update(tasks[t]["position"], tr[t])
            if args.env_name == 'AntVel-v1':
                teacher.task_generator.update(np.array([tasks[t]["velocity"]]), tr[t])

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)

        # Save tree
        # torch.save(tree, os.path.join(save_folder, 'tree-{0}.pt'.format(batch)))
        with open(os.path.join(save_folder,
                'tree-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(tree.state_dict(), f)


def eval(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
                                            'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
                                            '2DNavigation-v0'])

    # writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    log_folder = './logs/{0}'.format(args.output_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    if args.env_name == 'AntVel-v1':
        param_bounds = {"goal": [0, 3]}

    if args.env_name == 'AntPos-v0':
        param_bounds = {"x": [-3, 3],
                        "y": [-3, 3]}

    tree = TreeLSTM(args.tree_hidden_layer, len(param_bounds.keys()), args.cluster_0, args.cluster_1,
                    device=args.device)
    tree.to(args.device)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape) + args.tree_hidden_layer),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers, tree=tree)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape) + args.tree_hidden_layer),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers, tree=tree)
    policy.eval()
    tree.eval()


    all_tasks = []
    # torch.autograd.set_detect_anomaly(True)
    reward_list = []
    for batch in range(24):
        print("starting iteration {}".format(batch))
        policy.load_state_dict(torch.load(os.path.join(save_folder,
                                                       'policy-{0}.pt'.format(batch))))

        # tree.load_state_dict(torch.load(os.path.join(save_folder,
        #                        'tree-{0}.pt'.format(batch))))


        tasks = sampler.sample_tasks(args.meta_batch_size)

        all_tasks.append(tasks)
        # tasks = np.array(tasks)
        # tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        with open('./logs/{0}/task_list_eval.pkl'.format(args.output_folder), 'wb') as pf:
            pickle.dump(all_tasks, pf)

        print("evaluating...".format(batch))
        all_rewards = []
        for task in tasks:
            print(task["position"])
            episodes = sampler.sample(policy, task["position"])
        # print("training...".format(batch))


            # tr = [ep.rewards for ep in episodes]
            # tr = np.mean([torch.mean(torch.sum(rewards, dim=0)).item() for rewards in tr])
            all_rewards.append(total_rewards(episodes.rewards))

        reward_list.append(np.mean(all_rewards))



    with open('./logs/{0}/reward_list_eval.pkl'.format(args.output_folder), 'wb') as pf:
        pickle.dump(reward_list, pf)

    print(reward_list)




def eval(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
                                            'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
                                            '2DNavigation-v0'])

    # writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    log_folder = './logs/{0}'.format(args.output_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    if args.env_name == 'AntPos-v0':
        param_bounds = {"x": [-3, 3],
                        "y": [-3, 3]}

    tree = TreeLSTM(args.tree_hidden_layer, len(param_bounds.keys()), args.cluster_0, args.cluster_1, device=args.device)


    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    policy.eval()
    tree.eval()


    all_tasks = []
    # torch.autograd.set_detect_anomaly(True)
    reward_list = []
    for batch in range(args.num_batches + 1):
        print("starting iteration {}".format(batch))
        try:
            policy.load_state_dict(torch.load(os.path.join(save_folder,
                                                       'policy-{0}.pt'.format(batch))))
            tree = torch.load(os.path.join(save_folder,
                                                       'tree-{0}.pt'.format(batch)))
            tree.eval()
        except Exception:
            with open('./logs/{0}/reward_list_eval.pkl'.format(args.output_folder), 'wb') as pf:
                pickle.dump(reward_list, pf)

            print(reward_list)
            return

        # tree.load_state_dict(torch.load(os.path.join(save_folder,
        #                        'tree-{0}.pt'.format(batch))))


        tasks = sampler.sample_tasks(args.meta_batch_size)

        all_tasks.append(tasks)
        # tasks = np.array(tasks)
        # tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        with open('./logs/{0}/task_list_eval.pkl'.format(args.output_folder), 'wb') as pf:
            pickle.dump(all_tasks, pf)

        print("evaluating...".format(batch))
        all_rewards = []
        for task in tasks:
            print(task["position"])
            episodes = sampler.sample(policy, task, tree=tree)
        # print("training...".format(batch))


            # tr = [ep.rewards for ep in episodes]
            # tr = np.mean([torch.mean(torch.sum(rewards, dim=0)).item() for rewards in tr])
            all_rewards.append(total_rewards(episodes.rewards))

        reward_list.append(np.mean(all_rewards))



    with open('./logs/{0}/reward_list_eval.pkl'.format(args.output_folder), 'wb') as pf:
        pickle.dump(reward_list, pf)

    print(reward_list)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    parser.add_argument('--teacher', type=str, default='ALP-GMM',
        help='teacher variant')
    parser.add_argument('--nb_test_episodes', type=int, default=50,
        help='number of test tasks')
    parser.add_argument('--tree_hidden_layer', type=int, default=40,
                        help='size of treelstm hidden layer')
    parser.add_argument('--cluster_0', type=int, default=4,
                        help='number of clusters in first tree layer')
    parser.add_argument('--cluster_1', type=int, default=2,
                        help='number of clusters in second tree layer')
    parser.add_argument('--seed', type=float, default=2,
        help='teacher seed')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')
    parser.add_argument('--do-eval', action='store_true',
                        help='do eval')

    parser.add_argument('--do-eval', action='store_true',
                        help='do eval')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
                               if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    if not args.do_eval:
        main(args)
    else:
        eval(args)
