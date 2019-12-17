import gym
import torch
import multiprocessing as mp
import numpy as np

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env

class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        self._env = gym.make(env_name)

    def sample(self, policy, task, tree=None, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                input = torch.from_numpy(observations).to(device=device)

                if self.env_name == 'AntPos-v0':
                    _, embedding = tree.forward(torch.from_numpy(task["position"]).to(device=device))
                if self.env_name == 'AntVel-v1':
                    _, embedding = tree.forward(torch.from_numpy(np.array([task["velocity"]])).float().to(device=device))

                # print(input.shape)
                # print(embedding.shape)
                observations_tensor = torch.t(
                    torch.stack([torch.cat([torch.from_numpy(np.array(teo)).to(device=device), embedding[0]], 0) for teo in input], 1))

                actions_tensor = policy(observations_tensor, task=task, params=params, enhanced=False).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations_tensor.cpu().numpy(), actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
