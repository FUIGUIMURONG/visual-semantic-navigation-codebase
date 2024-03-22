from __future__ import print_function, division

import os
import random
import ctypes
import setproctitle
import time
import json
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils import flag_parser
from utils.flag_parser import parse_arguments
from utils.utils import ScalarMeanTracker, compute_loss, transfer_gradient_from_player_to_shared, reset_player, compute_single_spl_offline
from optimizer import SharedAdam
from models import BaseModel, GCN
from agent import NavAgent


def train_A3C_process(rank, args, shared_model, optimizer, res_queue, end_flag):
    setproctitle.setproctitle("A3C Training Agent: {}".format(rank))
    with open(os.path.join(args.exp_data_dir, "train_seen.json"), "r") as f:
        train_data = json.load(f)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    idx = [j for j in range(len(train_data))]
    random.shuffle(idx)

    if -1 in args.gpu_ids:
        args.device = torch.device("cpu")
    else:
        gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
        args.device = torch.device(f"cuda:{gpu_id}")
    
    if optimizer == None:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, shared_model.parameters()), lr = args.RL_lr)
    
    model = None
    assert (args.RL_model == 'BaseModel' or args.RL_model == 'GCN')
    if args.RL_model == 'BaseModel':
        model = BaseModel(args)
    elif args.RL_model == 'GCN':
        model = GCN(args)
    model = model.to(args.device)
    model.train()

    Agent = NavAgent(args, model)

    # ids = 0
    while not end_flag.value:
        # Get a new episode.
        player_start_time = time.time()        
        task_data = train_data[idx]
        total_reward = 0
        Agent.eps_len = 0
        Agent.new_episode(args, task_data)

        # Train on the new episode.
        while Agent.done == False:
            # Make sure model is up to date.
            Agent.sync_with_shared(shared_model)

            # Run episode for num_steps or until player is done.
            for i_step in range(args.num_steps):
                Agent.action(training = True)
                total_reward = total_reward + Agent.reward
                if Agent.done == True:
                    break
            
            Agent.model.zero_grad()
            loss = compute_loss(args, Agent)
            loss["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(Agent.model.parameters(), 100.0)
            transfer_gradient_from_player_to_shared(Agent, shared_model, args.gpu_ids)
            optimizer.step()
            
            if Agent.done == False:
                reset_player(Agent)
    
        # print("{} scene. total reward:{:0.5f}".format(ids, total_reward))

        for k in loss:
            loss[k] = loss[k].item()

        results = {
            "success": int(Agent.episode_successful),
            "total_reward": total_reward,
            "total_loss": loss["total_loss"],
            "policy_loss": loss["policy_loss"], 
            "value_loss": loss["value_loss"],
            "total_time": time.time() - player_start_time,
        }
        res_queue.put(results)
        reset_player(Agent)
        ids = (ids + 1) % len(train_data)


def train_A3C(args):
    setproctitle.setproctitle("A3C Train Manager for VSN")

    full_time = time.localtime(time.time())
    year, mon, day, hour, min_ = full_time.tm_year, full_time.tm_mon, full_time.tm_mday, full_time.tm_hour, full_time.tm_min
    tb_dir = os.path.join(args.tb_dir, 'RL_{}_{}_{}_{}_{}_{}'.format(year, mon, day, hour, min_, args.RL_model))
    if os.path.exists(tb_dir) == False:
        os.makedirs(tb_dir, exist_ok = True)
    writer = SummaryWriter(log_dir = os.path.join(tb_dir, 'RL_train'))
    args_str = {}
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
        args_str[k] = args.__dict__[k]
    with open(os.path.join(tb_dir, 'args_config.json'), 'w') as f:
        f.write(json.dumps(args_str, indent = 4))

    shared_model = None
    assert (args.RL_model == 'BaseModel' or args.RL_model == 'GCN')
    if args.RL_model == 'BaseModel':
        shared_model = BaseModel(args)
    elif args.RL_model == 'GCN':
        shared_model = GCN(args)
    
    if args.If_IL_pretrain == True:
        state_dict = torch.load(args.load_IL_path)
        shared_model.load_state_dict(state_dict)
    shared_model.share_memory()

    if args.shared == True:
        optimizer = SharedAdam(filter(lambda p: p.requires_grad, shared_model.parameters()), args)
        optimizer.share_memory()
    else:
        optimizer = None
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    assert (-1 in args.gpu_ids and len(args.gpu_ids) == 1) or (-1 not in args.gpu_ids and len(args.gpu_ids) > 0), "Invalid gpu_ids input!"
    if -1 not in args.gpu_ids:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn", force = True)

    processes = []
    end_flag = mp.Value(ctypes.c_bool, False)
    train_res_queue = mp.Queue()
    for rank in range(0, args.num_workers):
        p = mp.Process(
            target = train_A3C_process,
            args = (rank, args, shared_model, optimizer, train_res_queue, end_flag),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
    print("Train agents created.")

    train_total_ep = 0
    train_scalars = ScalarMeanTracker()
    try:
        while train_total_ep < args.max_RL_episode:
            train_result = train_res_queue.get()
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            if train_total_ep % args.n_record_RL == 0:
                writer.add_scalar("RL_train/RL_lr", optimizer.param_groups[0]['lr'], train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    writer.add_scalar("RL_train/" + k, tracked_means[k], train_total_ep)
                print('[episode {}] total loss:{:.5f} policy loss:{:.5f} value loss:{:.5f}'.format(train_total_ep, tracked_means['total_loss'], tracked_means['policy_loss'], tracked_means['value_loss']))
                print('[episode {}] total reward:{:.5f}'.format(train_total_ep, tracked_means['total_reward']))

            if train_total_ep % args.RL_save_episodes == 0:
                state_to_save = shared_model.state_dict()
                save_path = os.path.join(tb_dir, 'epoch_{}.pth'.format(train_total_ep))
                torch.save(state_to_save, save_path)

    finally:
        writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()


def test_A3C(args, split, setting):
    assert (split == 'val' or split == 'test')
    assert (setting == 'seen' or setting == 'unseen' or setting == 'all')
    with open(os.path.join(args.exp_data_dir, f"{split}_{setting}.json"), "r") as f:
        test_data = json.load(f)

    if -1 in args.gpu_ids:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpu_ids[0]}")

    model = None
    assert (args.RL_model == 'BaseModel' or args.RL_model == 'GCN')
    if args.RL_model == 'BaseModel':
        model = BaseModel(args)
    elif args.RL_model == 'GCN':
        model = GCN(args)
    state_dict = torch.load(args.test_RL_load_model)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    Agent = NavAgent(args, model)

    test_success = []
    test_spl = []
    test_path_length = []
    for i_task, task_data in enumerate(test_data):       
        total_reward = 0
        Agent.eps_len = 0
        Agent.new_episode(args, task_data)

        for i_step in range(args.max_episode_length):
            Agent.action(training = False)
            total_reward = total_reward + Agent.reward
            if Agent.done == True:
                break
        
        reset_player(Agent)
        test_success.append(int(Agent.episode_successful))
        init_start = Agent.controller.get_state_from_str(
            x = float(task_data["init"].split("|")[0]), 
            z = float(task_data["init"].split("|")[1]),
            rotation = float(task_data["init"].split("|")[2]),
            horizon = float(task_data["init"].split("|")[3])
        )
        single_spl, shortest_step_len, path_length = compute_single_spl_offline(Agent, init_start, task_data["target"])
        print(f"ep_len:{Agent.eps_len} reward:{total_reward} success:{Agent.episode_successful}")
        test_spl.append(single_spl)
        test_path_length.append(path_length)
        print(f"{i_task} episode finished.")
    print(f"success list:{test_success} spl list:{test_spl} length list:{test_path_length}")

    success_all = 0.0
    spl_all = 0.0
    success_5 = 0.0
    spl_5 = 0.0
    num_5 = 0
    for i_task in range(len(test_success)):
        success_all += test_success[i_task]
        spl_all += test_success[i_task] * test_spl[i_task]
        if test_path_length[i_task] >= 5:
            success_5 += test_success[i_task]
            spl_5 += test_success[i_task] * test_spl[i_task]
            num_5 += 1
    success_all = success_5 / len(test_data)
    spl_all = spl_all / len(test_data)
    success_5 = success_5 / num_5
    spl_5 = spl_5 / num_5

    res_dict = {
        "success_all": success_all, 
        "success_larger5": success_5,
        "spl_all": spl_all,
        "spl_larger5": spl_5
    }
    if os.path.exists(args.test_out_dir) == False:
        os.makedirs(args.test_out_dir, exist_ok = True)
    res_sav_path = os.path.join(args.test_out_dir, f"res_A3C_{args.RL_model}_{split}_{setting}.json")
    with open(res_sav_path, "w") as f:
        f.write(json.dumps(res_dict, indent = 4))


if __name__ == "__main__":
    args = parse_arguments()
    # train_A3C(args)
    test_A3C(args, "test", "seen")
    # test_A3C(args, "test", "unseen")
    # test_A3C(args, "test", "all")