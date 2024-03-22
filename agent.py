from __future__ import division

import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import torchvision.transforms as transforms
import ai2thor
from ai2thor.controller import Controller
import h5py
import math
from controller_offline import OfflineController
from dataset.constants import ALL_ACTION_LIST, GOAL_SUCCESS_REWARD, STEP_PENALTY, ACTION_FAIL_PENALTY, STOP_NOT_SUCCESS
from utils.utils import get_glove_embeddings


class NavAgent:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.controller = OfflineController(
            args = args,
            offline_data_dir = args.offline_data_dir,
            grid_file_name = args.grid_file_name,
            graph_file_name = args.graph_file_name,
            input_file_name = args.input_file_name,
            object_metadata_file_name = args.object_metadata_file_name
        )
        # self.glove_embeddings = h5py.File(args.glove_file, "r")
        self.glove_embeddings = get_glove_embeddings(args)
        self.actions_list = ALL_ACTION_LIST
        self.scene = None
        self.target = None
        self.target_glove_embed = None
        self.device = args.device
    
        self.prev_frame = None
        self.current_frame = None
        self.hidden = None
        self.hidden_state_dim = args.hidden_state_dim
        self.last_action_probs = None
        self.action_space = args.action_space
        self.max_episode_length = args.max_episode_length
        
        self.eps_len = 0
        self.failed_action_count = 0
        self.reward = 0
        self.values = []
        self.probs = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.last_actions = []
        self.memory = []
        self.done_action_probs = []
        self.done_action_targets = []
        
        self.done = False
        self.action_successful = False
        self.episode_successful = False
        self.max_length = False
        self.success_target_id = None
        self.candidate_target = []
    

    def new_episode(self, args, task_data):
        self.clear_states()
        self.scene = task_data["scene"]
        self.target = task_data["target"]
        # self.target_glove_embed = torch.FloatTensor(self.glove_embeddings[self.target][:]).to(args.device)
        self.target_glove_embed = torch.FloatTensor(self.glove_embeddings[self.target]).to(args.device)
        init_pos = task_data["init"]
        self.controller.reset(self.scene)
        self.controller.teleport(init_pos)

        self.prev_frame = None
        self.current_frame = self.controller.get_input()
        self.hidden = None
        self.last_action_probs = None
        self.failed_action_count = 0

        self.done = False
        self.action_successful = False
        self.episode_successful = False
        self.max_length = False
        self.success_target_id = None
        self.candidate_target = []
        self.reset_hidden()

        all_objects = self.controller.all_objects()
        for obj in all_objects:
            if obj.split('|')[0] == self.target:
                self.candidate_target.append(obj)


    def sync_with_shared(self, shared_model):
        self.model.load_state_dict(shared_model.state_dict())


    def _increment_episode_length(self):
        self.eps_len += 1
        if self.eps_len >= self.max_episode_length:
            self.max_length = True
            self.done = True
        else:
            self.max_length = False


    def step(self, action_as_int):
        """
        reward - The reward of this step
        done - If the episode is done. If perform "Stop" or the total number of steps execeed the max_episode_length, done is True.
        action_was_success - If the performed action is successful.
        """
        action = self.actions_list[action_as_int]
        pre_pos = self.controller.last_event.metadata["agent"]["position"]

        if action != "Stop":
            self.controller.step(action)
        
        cur_pos = self.controller.last_event.metadata['agent']['position']
        tar_pos = self.closest_obj_pos()
        pre_dis = math.sqrt((pre_pos['x'] - tar_pos['x']) ** 2 + (pre_pos['z'] - tar_pos['z']) ** 2)
        cur_dis = math.sqrt((cur_pos['x'] - tar_pos['x']) ** 2 + (cur_pos['z'] - tar_pos['z']) ** 2)
        difference_dis = pre_dis - cur_dis
        reward = STEP_PENALTY + difference_dis
        
        done = False
        episode_successful = False
        
        if self.args.strict_done == True:
            if action == "Stop":
                done = True
                action_successful = False
                for id_ in self.candidate_target:
                    if self.controller.object_is_visible(id_) == True:
                        reward = GOAL_SUCCESS_REWARD
                        action_successful = True
                        episode_successful = True
                        self.success_target_id = id_
                        break
                if episode_successful == False:
                    reward += STOP_NOT_SUCCESS
            else:
                action_successful = self.controller.last_action_success
        else:
            done = False
            action_successful = False
            for id_ in self.candidate_target:
                if self.controller.object_is_visible(id_) == True:
                    reward = GOAL_SUCCESS_REWARD
                    action_successful = True
                    episode_successful = True
                    done = True
                    self.success_target_id = id_
                    break
            if action == "Stop":
                # done = True
                if episode_successful == False:
                    done = False
                    reward += STOP_NOT_SUCCESS
        
        if action != "Stop" and action_successful == False:
            self.failed_action_count += 1
            reward += ACTION_FAIL_PENALTY

        return reward, done, action_successful, episode_successful


    def action(self, training):
        if training == True:
            self.model.train()
        else:
            self.model.eval()

        current_frame, model_output = self.generate_action()
        actor_out, critic_out, (hs, cs) = model_output
        # print(hs.shape, cs.shape)
        # hs = hs.unsqueeze(1)
        # cs = cs.unsqueeze(1)
        self.hidden = (hs, cs)
        prob = F.softmax(actor_out.squeeze(1), dim = 1)
        self.last_action_probs = prob.unsqueeze(1)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = F.log_softmax(actor_out.squeeze(1), dim = 1)

        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob[:, action.item()]

        self.reward, self.done, self.action_successful, self.episode_successful = self.step(action.item())

        self.probs.append(prob)
        self.entropies.append(entropy)
        self.values.append(critic_out.squeeze(1))
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.actions.append(action)
        self.prev_frame = current_frame
        self.current_frame = self.controller.get_input()

        self._increment_episode_length()
        if self.max_length == True:
            self.done = True

        return critic_out, prob, action


    def clear_states(self):
        self.reward = 0
        self.failed_action_count = 0
        self.values = []
        self.probs = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.last_actions = []
        self.memory = []


    def generate_action(self):
        target = self.target_glove_embed.unsqueeze(0).unsqueeze(0)
        if not self.current_frame.any():
            self.current_frame = self.controller.get_input()
        
        if self.args.features_or_images == False:
            input_ = self.preprocess_frame(self.current_frame).unsqueeze(1).to(self.device)
        else:
            input_ = torch.FloatTensor(self.current_frame).unsqueeze(1).to(self.device)
        
        if self.last_action_probs == None:
            self.last_action_probs = torch.zeros((1, 1, self.action_space)).to(self.device)
        # print('generate action shape:', self.last_action_probs.shape)

        if self.hidden == None:
            self.hidden = (
                torch.zeros(1, 1, self.hidden_state_dim).to(self.device),
                torch.zeros(1, 1, self.hidden_state_dim).to(self.device),
            )
        
        actor_out, critic_out, (hs, cs) = self.model(target, input_, self.last_action_probs, self.hidden[0], self.hidden[1])
        return self.current_frame, (actor_out, critic_out, (hs, cs))
    

    def closest_obj_pos(self):
        cur_pos = self.controller.last_event.metadata["agent"]["position"]
        dis = float("inf")
        pos = None
        for id_ in self.candidate_target:
            x, y, z = float(id_.split('|')[1]), float(id_.split('|')[2]), float(id_.split('|')[3])
            temp_dis = math.sqrt((cur_pos['x'] - x) ** 2 + (cur_pos['z'] - z) ** 2)
            if temp_dis < dis:
                dis = temp_dis
                pos = {'x': x, 'y': y, 'z': z}
        return pos


    def preprocess_frame(self, frame):
        prepro_frame = torch.from_numpy(frame.copy()).float()
        prepro_frame = prepro_frame.permute((2, 0, 1))
        prepro_frame = transforms.Resize((224, 224))(prepro_frame)
        prepro_frame = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(prepro_frame)

        return prepro_frame


    def reset_hidden(self):
        self.hidden = (
                torch.zeros(1, 1, self.hidden_state_dim).to(self.device),
                torch.zeros(1, 1, self.hidden_state_dim).to(self.device),
        )
        self.last_action_probs = torch.zeros((1, 1, self.action_space)).to(self.device)


    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

