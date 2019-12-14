import torch
import torch.nn as nn

from torch.autograd import Variable

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class TreeLSTM(nn.Module):
    def __init__(self, tree_hidden_dim, input_dim, cluster_layer_0, cluster_layer_1, device=torch.device('cpu')):
        super(TreeLSTM, self).__init__()
        self.input_dim = input_dim
        self.tree_hidden_dim = tree_hidden_dim
        self.leaf_i, self.leaf_o, self.leaf_u = [], [], []

        self.cluster_layer_0 = cluster_layer_0
        self.cluster_layer_1 = cluster_layer_1

        for i in range(self.cluster_layer_0):
            # w = torch.randn((input_dim, tree_hidden_dim), requires_grad=True, device=device)
            # self.leaf_weight_u.append(w)
            # w = torch.randn((1, tree_hidden_dim), requires_grad=True, device=device)
            # self.leaf_bias_u.append(w)
            self.leaf_u.append(
                nn.Linear(input_dim, tree_hidden_dim))
        self.leaf_u = nn.ModuleList(self.leaf_u)
        self.leaf_u.apply(weight_init)

        self.no_leaf_i, self.no_leaf_o, self.no_leaf_u, self.no_leaf_f = [], [], [], []
        for i in range(self.cluster_layer_1):
            # if True:
            #     self.no_leaf_weight_i.append(
            #         torch.randn((tree_hidden_dim, 1), requires_grad=True, device=device))
            # # elif FLAGS.tree_type==2:
            # #     self.no_leaf_weight_i.append(
            # #         tf.get_variable(name='{}_no_leaf_weight_i'.format(i), shape=(1, tree_hidden_dim)))
            # self.no_leaf_weight_u.append(
            #     torch.randn((tree_hidden_dim, tree_hidden_dim), requires_grad=True, device=device))
            #
            # self.no_leaf_bias_i.append(torch.randn((1, 1), requires_grad=True, device=device))
            # self.no_leaf_bias_u.append(torch.randn((1, tree_hidden_dim), requires_grad=True, device=device))
            self.no_leaf_i.append(
                        nn.Linear(tree_hidden_dim, 1))
            self.no_leaf_u.append(
                        nn.Linear(tree_hidden_dim, tree_hidden_dim))
        self.no_leaf_i = nn.ModuleList(self.no_leaf_i)
        self.no_leaf_u = nn.ModuleList(self.no_leaf_u)

        self.no_leaf_i.apply(weight_init)
        self.no_leaf_u.apply(weight_init)


        self.root_leaf_u = nn.Linear(tree_hidden_dim, tree_hidden_dim)
        weight_init(self.root_leaf_u)
        # self.root_weight_u = torch.randn((tree_hidden_dim, tree_hidden_dim), requires_grad=True, device=device)
        #
        # self.root_bias_u = torch.randn((1, tree_hidden_dim), requires_grad=True, device=device)



        cluster_centers = []
        for i in range(self.cluster_layer_0):
            w = torch.empty(1, input_dim)
            nn.init.xavier_uniform_(w)
            cluster_centers.append(nn.Parameter(w))

        self.cluster_center = nn.ParameterList(cluster_centers)


    def forward(self, inputs):

        sigma = 5

        for idx in range(self.cluster_layer_0):
            x = inputs - self.cluster_center[idx]
            if idx == 0:
                all_value = torch.exp(-torch.sum(torch.mul(x, x)) / (2.0*sigma))
            else:
                all_value += torch.exp(-torch.sum(torch.mul(x, x)) / (2.0*sigma))

        c_leaf = []
        for idx in range(self.cluster_layer_0):
            x = inputs - self.cluster_center[idx]
            assignment_idx = torch.exp(-torch.sum(torch.mul(x, x)) / (2.0*sigma)) / all_value
            value_u = torch.tanh(self.leaf_u[idx](inputs))
            value_c = assignment_idx * value_u
            value_c.unsqueeze_(0)
            c_leaf.append(value_c)
        print(c_leaf)
        c_no_leaf = []
        for idx in range(self.cluster_layer_0):
            input_gate = []
            for idx_layer_1 in range(self.cluster_layer_1):
                if True:
                    input_gate.append(
                        self.no_leaf_i[idx_layer_1](c_leaf[idx]))
                # elif FLAGS.tree_type == 2:
                #     input_gate.append(
                #         -(tf.reduce_sum(tf.square(c_leaf[idx] - self.no_leaf_weight_i[idx_layer_1]), keepdims=True) +
                #           self.no_leaf_bias_i[idx_layer_1]) / (
                #             2.0))
            print(input_gate)
            input_gate = torch.nn.functional.softmax(torch.cat(input_gate, 0), dim=0)
            c_no_leaf_temp = []
            for idx_layer_1 in range(self.cluster_layer_1):
                no_leaf_value_u = torch.tanh(self.no_leaf_u[idx_layer_1](c_leaf[idx]))
                c_no_leaf_temp.append(input_gate[idx_layer_1] * no_leaf_value_u)
            c_no_leaf.append(torch.cat(c_no_leaf_temp, 0))
            # print()

        c_no_leaf = torch.stack(c_no_leaf, 0)
        # print(c_no_leaf.shape)
        c_no_leaf = c_no_leaf.permute(1, 0, 2)
        c_no_leaf = torch.sum(c_no_leaf, dim=1, keepdim=True)

        root_c = []

        for idx in range(self.cluster_layer_1):
            root_c.append(torch.tanh(self.root_leaf_u(c_no_leaf[idx])))

        root_c = torch.sum(torch.cat(root_c, 0), dim=0, keepdim=True)

        return root_c, root_c

if __name__=="__main__":
    k = TreeLSTM(4, 2, 3, 3)
    print(k.forward(torch.tensor([1., 2.])))
    for param_tensor in k.state_dict():
        print(param_tensor, "\t", k.state_dict()[param_tensor].size())
    # [-2.0737,  0.4270,  1.6740, -2.9784]
    # tree = torch.load("../test/tree.pkl")
    # tree.eval()
    # print(tree.forward(torch.tensor([1., 2.])))
