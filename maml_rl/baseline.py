import torch
import torch.nn as nn

class LinearFeatureBaseline(nn.Module):
    """Linear baseline based on handcrafted features, as described in [1] 
    (Supplementary Material 2).

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, input_size, tree, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.linear = nn.Linear(self.feature_size, 1, bias=False)
        self.tree = tree
        self.linear.weight.data.zero_()

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes, task):
        with torch.no_grad():
            ones = episodes.mask.unsqueeze(2)
            observations = episodes.observations * ones
            cum_sum = torch.cumsum(ones, dim=0) * ones
            al = cum_sum / 100.0

            embedding = self.tree(torch.from_numpy(task))
            embedding = embedding.clone()
            # if self.env_name == 'AntVel-v1':
            #     _, embedding = self.tree(torch.from_numpy(np.array([task["velocity"]])))

            # print(input.shape)
            if len(observations.shape) == 2:
                observations = torch.t(
                    torch.stack([torch.cat([teo.clone(), embedding[0].clone()], 0).clone() for teo in observations],
                                1).clone()).clone()
            if len(observations.shape) == 3:
                observations = torch.stack([torch.t(
                    torch.stack([torch.cat([teo.clone(), embedding[0].clone()], 0).clone() for teo in tei], 1).clone())
                    for tei in observations], 1).clone().permute(1, 0, 2)

            f = torch.cat([observations, observations ** 2,
                al, al ** 2, al ** 3, ones], dim=2)

        # print(f.shape)

        return f

    def fit(self, episodes, task):
        # sequence_length * batch_size x feature_size
        # print(episodes.shape)
        featmat = self._feature(episodes, task).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        returns = episodes.returns.view(-1, 1)

        reg_coeff = self._reg_coeff
        eye = torch.eye(self.feature_size, dtype=torch.float32,
            device=self.linear.weight.device)
        for _ in range(5):
            try:
                coeffs, _ = torch.lstsq(
                    torch.matmul(featmat.t(), returns),
                    torch.matmul(featmat.t(), featmat) + reg_coeff * eye
                )
                break
            except RuntimeError:
                reg_coeff += 10
        else:
            raise RuntimeError('Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.linear.weight.data = coeffs.data.t()

    def forward(self, episodes, task):
        features = self._feature(episodes, task)
        return self.linear(features)
