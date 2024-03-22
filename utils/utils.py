import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import math
import h5py


def get_glove_embeddings(args):
    glove_embedding_file = h5py.File(args.glove_file, "r")
    glove_keys = list(glove_embedding_file.keys())
    final_glove_embeddings = {}
    for key in glove_keys:
        final_glove_embeddings[key] = np.array(glove_embedding_file[key][:])
    glove_embedding_file.close()
    return final_glove_embeddings


def a3c_loss(args, player):
    R = torch.zeros(1, 1)
    if not player.done:
        _, output = player.generate_action()
        _, critic_out, _ = output
        R = critic_out.data
    R = R.to(args.device)

    player.values.append(Variable(R))
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1).to(args.device)

    R = Variable(R)
    for i in reversed(range(len(player.rewards))):
        R = args.gamma * R + player.rewards[i]
        advantage = R - player.values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)
        delta_t = player.rewards[i] + args.gamma * player.values[i + 1].data - player.values[i].data
        gae = gae * args.gamma * args.tau + delta_t
        policy_loss = policy_loss - player.log_probs[i] * Variable(gae) - args.beta * player.entropies[i]

    return policy_loss, value_loss


def compute_loss(args, player):
    policy_loss, value_loss = a3c_loss(args, player)
    total_loss = policy_loss + 0.5 * value_loss
    # print(f"total loss:{total_loss} policy loss:{policy_loss} value loss:{value_loss}")
    return dict(total_loss = total_loss, policy_loss = policy_loss, value_loss = value_loss)


def transfer_gradient_from_player_to_shared(player, shared_model, gpu_ids):
    """
    Transfer the gradient from the player's model to the shared model
    """
    for param, shared_param in zip(player.model.parameters(), shared_model.parameters()):
        if shared_param.requires_grad:
            # print(f'shared_param:{shared_param._grad} param:{param.grad}')
            # if param.grad is None:
            #     shared_param._grad = torch.zeros(shared_param.shape)
            # elif -1 in gpu_ids:
            #     shared_param._grad = param.grad
            # else:
            #     shared_param._grad = param.grad.cpu()
            
            if shared_param.grad is not None and -1 in gpu_ids:
                return
            elif -1 in gpu_ids:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()
                

def reset_player(player):
    player.clear_states()
    player.repackage_hidden()


def norm_col_init(weights, std = 1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim = True))
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    
    # if isinstance(m, (nn.Linear, nn.Conv2d)):
    #     # print('kaiming_init:', m)
    #     init.kaiming_normal_(m.weight)
    #     if m.bias is not None:
    #         m.bias.data.fill_(0)
    # elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
    #     m.weight.data.fill_(1)
    #     if m.bias is not None:
    #         m.bias.data.fill_(0)


def compute_single_spl_offline(player, start_state, target):
    """
    return results:
    single_spl - the spl value of single sample
    shortest_path_len - the shortest step length from the start state to the target object type
    path_length - the shortest path length from the start state to the target object type
    """
    # shortest_path_len = 10000
    _, shortest_path_len, path_plan = player.controller.shortest_path_to_target_type(start_state, target, True)
    
    num_mov_action = 0
    for act in path_plan:
        if act in ["MoveAhead", "MoveRight", "MoveLeft"]:
            num_mov_action += 1
    path_length = num_mov_action * 0.25

    if not player.episode_successful:
        return 0, shortest_path_len, path_length

    if shortest_path_len < 10000:
        return shortest_path_len / float(player.eps_len), shortest_path_len, path_length

    return 0, shortest_path_len, path_length


class ScalarMeanTracker(object):
    def __init__(self) -> None:
        self._sums = {}
        self._counts = {}

    def add_scalars(self, scalars):
        for k in scalars:
            if k not in self._sums:
                self._sums[k] = scalars[k]
                self._counts[k] = 1
            else:
                self._sums[k] += scalars[k]
                self._counts[k] += 1

    def pop_and_reset(self):
        means = {k: self._sums[k] / self._counts[k] for k in self._sums}
        self._sums = {}
        self._counts = {}
        return means
