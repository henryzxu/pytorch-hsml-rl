import numpy as np
from teachDRL.teachers.algos.alp_gmm import ALPGMM
from teachDRL.teachers.utils.dataset import BufferedDataset


class EmpiricalSkillComputer:
    def __init__(self, task_size, skill_size, max_size=None, buffer_size=500):
        self.skill_knn = BufferedDataset(skill_size, task_size, buffer_size=buffer_size, lateness=0, max_size=max_size)

    def compute_skill_isolation(self, task, skill):
        # Add to database
        self.skill_knn.add_xy(skill, task)

    def get_isolated_skill(self):
        farthest_val = -np.inf
        farthest_dists = None
        farthest_skill = None
        for skill in self.skill_knn.iter_x():
            nearest_skill, idx = self.skill_knn.nn_x(skill)
            dist_vec = np.power(skill - nearest_skill, 2)
            dist = np.sum(dist_vec)
            if dist > farthest_val:
                farthest_dists = dist_vec
                farthest_skill = skill
        return farthest_skill, farthest_dists

class SKALPGMM(ALPGMM):
    def __init__(self, mins, maxs, skill_size, seed=None, params=dict()):
        super().__init__(mins, maxs, seed, params)

        alp_max_size = None if "alp_max_size" not in params else params["alp_max_size"]
        alp_buffer_size = 500 if "alp_buffer_size" not in params else params["alp_buffer_size"]

        self.skill_computer = EmpiricalSkillComputer(len(mins), skill_size, max_size=alp_max_size, buffer_size=alp_buffer_size)
        self.alp_task_ratio = 0.8 if "alp_task_ratio" not in params else params["alp_task_ratio"]
        self.isolation_scale_factor = 0.5 if "alp_task_ratio" not in params else params["alp_task_ratio"]


    def update(self, task, student_return):
        reward, skill_vector = student_return
        super().update(task, reward)
        self.skill_computer.compute_skill_isolation(task, skill_vector)



    def sample_task(self):
        if (len(self.tasks) < self.nb_random) or (np.random.random() < self.alp_task_ratio):
            new_task = super().sample_task()
        else:
            farthest_skill, farthest_dists = self.skill_computer.get_isolated_skill()
            new_task = np.random.multivariate_normal(farthest_skill, np.diag(farthest_dists * self.isolation_scale_factor))

        return new_task

    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict