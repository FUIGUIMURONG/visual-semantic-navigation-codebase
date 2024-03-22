import os
import json
import torch
import h5py
import sys
import random
import math
import importlib
import pickle
import numpy as np
from constants import ACTION_LIST, ALL_ACTION_LIST, ALL_ACTION_TO_ID


TARGET_OBJECTS = [
    "Pillow",
    "Laptop",
    "Television",
    "GarbageCan", 
    "Box"
]

SEEN_OBJECTS = [
    "Pillow",
    "Laptop",
    "Television"
]

UNSEEN_OBJECTS = [
    "GarbageCan", 
    "Box"
]

train_scene = [f"FloorPlan{ids}" for ids in range(221, 228)]
val_scene = ["FloorPlan228"]
test_scene = ["FloorPlan229"]


def generate_split_data(split, per_init = 30, seen_num_all = 620, unseen_num_all = 380, seen_num_larger5 = 310, unseen_num_larger5 = 190):
    assert (split == "train") or (split == "val") or (split == "test"), "Invalid Split Name."
    scene_list = eval(f"{split}_scene")   
    print("scene_list:", scene_list)
    
    # seen_data = []
    # unseen_data = []
    seen_larger5 = []
    seen_less5 = []
    unseen_larger5 = []
    unseen_less5 = []
    for scene in scene_list:
        data_info_dir = f"./scene_data/{scene}"
        with open(os.path.join(data_info_dir, "grid.json"), "r") as f:
            reachable_pos = json.load(f)['grid']
        with open(os.path.join(data_info_dir, "objects_where_visible.json"), "r") as f:
            objects_where_visible = json.load(f)
        
        all_objects = objects_where_visible.keys()
        for obj in all_objects:
            seen_init = []
            unseen_init = []
            if obj.split("|")[0] in TARGET_OBJECTS:
                if obj.split("|")[0] in SEEN_OBJECTS:
                    # target_data_list = seen_data
                    target_init = seen_init
                    target_larger5_data = seen_larger5
                    target_less5_data = seen_less5
                else:
                    # target_data_list = unseen_data
                    target_init = unseen_init
                    target_larger5_data = unseen_larger5
                    target_less5_data = unseen_less5
                
                idd = 0
                while idd < per_init:
                    rand_init = random.choice(reachable_pos)
                    obj_x, obj_z = float(obj.split("|")[1]), float(obj.split("|")[3])
                    init_x, init_z = float(rand_init.split("|")[0]), float(rand_init.split("|")[1])
                    if math.sqrt((init_x - obj_x) ** 2 + (init_z - obj_z) ** 2) <= 1.0:
                        continue
                    if rand_init not in target_init:
                        target_init.append(rand_init)
                        data_dict = {
                            "scene": scene,
                            "target": obj.split("|")[0],
                            "init": rand_init,
                        }
                        if math.sqrt((init_x - obj_x) ** 2 + (init_z - obj_z) ** 2) < 5.0:
                            target_less5_data.append(data_dict)
                        elif math.sqrt((init_x - obj_x) ** 2 + (init_z - obj_z) ** 2) >= 5.0:
                            target_larger5_data.append(data_dict)
                        # target_data_list.append(data_dict)
                        idd += 1
                print(f"{scene} {obj} finished.")
    print(f"seen <5 data:{len(seen_less5)} seen >=5 data:{len(seen_larger5)}  unseen <5 data:{len(unseen_less5)} unseen >=5 data:{len(unseen_larger5)}.")
    
    if len(seen_larger5) > seen_num_larger5:
        final_seen_larger5 = random.sample(seen_larger5, seen_num_larger5)
    else:
        final_seen_larger5 = seen_larger5
    if len(seen_less5) > seen_num_all - seen_num_larger5:
        final_seen_less5 = random.sample(seen_less5, seen_num_all - seen_num_larger5)
    else:
        final_seen_less5 = seen_less5
    
    if len(unseen_larger5) > unseen_num_larger5:
        final_unseen_larger5 = random.sample(unseen_larger5, unseen_num_larger5)
    else:
        final_unseen_larger5 = unseen_larger5
    if len(unseen_less5) > unseen_num_all - unseen_num_larger5:
        final_unseen_less5 = random.sample(unseen_less5, unseen_num_all - unseen_num_larger5)
    else:
        final_unseen_less5 = unseen_less5
    final_seen_data = final_seen_larger5 + final_seen_less5
    final_unseen_data = final_unseen_larger5 + final_unseen_less5

    print(f"[final] seen <5 data:{len(final_seen_less5)} seen >=5 data:{len(final_seen_larger5)}  unseen <5 data:{len(final_unseen_less5)} unseen >=5 data:{len(final_unseen_larger5)}.")
    print(f"[final] seen data: {len(final_seen_data)} final unseen data: {len(final_unseen_data)}.")
    final_all_data = final_seen_data + final_unseen_data
    
    data_split_dir = "./data_split"
    if os.path.exists(data_split_dir) == False:
        os.makedirs(data_split_dir, exist_ok = True)
    with open(os.path.join(data_split_dir, f"{split}_seen.json"), "w") as f:
        f.write(json.dumps(final_seen_data, indent = 4))
    with open(os.path.join(data_split_dir, f"{split}_unseen.json"), "w") as f:
        f.write(json.dumps(final_unseen_data, indent = 4))
    with open(os.path.join(data_split_dir, f"{split}_all.json"), "w") as f:
        f.write(json.dumps(final_all_data, indent = 4))


def get_next_state(action, state, grid_size = 0.25):
    x, z, rotation = float(state.split('|')[0]), float(state.split('|')[1]), int(state.split('|')[2])
    if action == "MoveAhead":
        if rotation == 0:
            z += grid_size
        elif rotation == 90:
            x += grid_size
        elif rotation == 180:
            z -= grid_size
        elif rotation == 270:
            x -= grid_size
    elif action == "MoveRight":
        if rotation == 0:
            x += grid_size
        elif rotation == 90:
            z -= grid_size
        elif rotation == 180:
            x -= grid_size
        elif rotation == 270:
            z += grid_size
    elif action == "MoveLeft":
        if rotation == 0:
            x -= grid_size
        elif rotation == 90:
            z += grid_size
        elif rotation == 180:
            x += grid_size
        elif rotation == 270:
            z -= grid_size
    elif action == "RotateRight":
        rotation = (rotation + 90) % 360
    elif action == "RotateLeft":
        rotation = (rotation - 90 + 360) % 360
    return "{:0.2f}|{:0.2f}|{:d}|0".format(x, z, round(rotation))


def optimal_plan(source_state, path):
    state = source_state
    actions = []
    i = 1
    while i < len(path):
        for a in ACTION_LIST:
            next_state = get_next_state(a, state)
            if next_state == path[i]:
                actions.append(a)
                i += 1
                state = next_state
                break
    return actions


def generate_metadata_expert_demonstration_IL(per_task = 5):
    train_data_path = './data_split/train_seen.json'
    with open(train_data_path, 'r') as f:
        train_data_dict = json.load(f)
    
    expert_demonstration = []
    for idx, data_dict in enumerate(train_data_dict):
        scene = data_dict['scene']
        target = data_dict['target']
        init = data_dict['init']
        scene_data_dir = f'./scene_data/{scene}'
        with open(os.path.join(scene_data_dir, 'graph.json'), 'r') as f:
            graph_json = json.load(f)
        scene_graph = importlib.import_module("networkx.readwrite").node_link_graph(graph_json).to_directed()

        with open(os.path.join(scene_data_dir, 'objects_where_visible.json'), 'r') as f:
            objects_where_visible = json.load(f)
        
        min_dis = 1000
        min_target_id = None
        for obj in objects_where_visible.keys():
            if obj.split("|")[0] == target:
                cur_x, cur_z = float(init.split("|")[0]), float(init.split("|")[1])
                tar_x, tar_z = float(obj.split("|")[1]), float(obj.split("|")[3])
                dis = math.sqrt((cur_x - tar_x) ** 2 + (cur_z - tar_z) ** 2)
                if dis < min_dis:
                    min_dis = dis
                    min_target_id = obj
        
        visible_pos = objects_where_visible[min_target_id]

        # tar_x, tar_z = float(target.split("|")[1]), float(target.split("|")[3])
        # dis_list = []
        # for v_p in visible_pos:
        #     v_x, v_z = float(v_p.split("|")[0], v_p.split("|")[1])
        #     dis = math.sqrt((v_x - tar_x) ** 2 + (v_z - tar_z) ** 2)
        #     dis_dict = {
        #         'pos': v_p,
        #         'dis': dis
        #     }
        #     dis_list.append(dis_dict)
        # dis_sort_list = sorted(dis_list, key = lambda e: e.__getitem__('dis'))

        record_shortest_path = []
        for v_p in visible_pos:
            shortest_path = importlib.import_module("networkx").shortest_path(scene_graph, init, v_p)
            actions = optimal_plan(init, shortest_path) + ['Stop']
            path_dict = {
                'scene': scene,
                'target': target,
                'init_pos': init,
                'target_pos': v_p,
                'actions': actions,
                'path_length': len(actions),
                'paths': shortest_path
            }
            record_shortest_path.append(path_dict)
        sorted_shortest_path = sorted(record_shortest_path, key = lambda e: e.__getitem__('path_length'))
        expert_demonstration += sorted_shortest_path[:per_task]
        print(f'{idx}-th training task finished.')
    print(f'num of expert demonstration:{len(expert_demonstration)}')

    with open('./data_split/expert_demonstration.json', 'w') as f:
        f.write(json.dumps(expert_demonstration, indent = 4))


def get_expert_demonstration_TF():
    exp_demon_dir = './data_split/expert_demonstration.json'
    with open(exp_demon_dir, 'r') as f:
        demonstrations = json.load(f)
    scene_dir = './scene_data/'
    glove_embeddings = h5py.File('./glove_map300d.hdf5', "r")

    expert_demonstration = []
    for idm, demon in enumerate(demonstrations):
        scene = demon['scene']
        cur_target = demon['target']
        init_pos = demon['init_pos']
        target_pos = demon['target_pos']
        actions = demon['actions']
        paths = demon['paths']
        scene_data_dir = os.path.join(scene_dir, scene)
        cur_images = h5py.File(os.path.join(scene_data_dir, 'images.h5'), 'r')
        cur_features = h5py.File(os.path.join(scene_data_dir, 'resnet18_features.h5'), 'r')

        num_demon_actions = len(actions)
        item_targets = []
        item_images = []
        item_features = []
        item_action_probs = []
        item_prev_action_probs = []
        item_actions = []
        for i_demon in range(num_demon_actions):
            pre_state = paths[i_demon]
            item_images.append(cur_images[pre_state][:])
            item_features.append(cur_features[pre_state][:].squeeze(0))
            item_targets.append(glove_embeddings[cur_target][:])
            
            cur_action_idx = ALL_ACTION_TO_ID[actions[i_demon]]
            cur_action_probs = np.zeros([len(ALL_ACTION_LIST)])
            cur_action_probs[cur_action_idx] = 1
            
            if i_demon == 0:
                prev_action_probs = np.zeros([len(ALL_ACTION_LIST)])
                prev_action_probs[:] = float(1.0 / len(ALL_ACTION_LIST))
            else:
                prev_action_idx = ALL_ACTION_TO_ID[actions[i_demon - 1]]
                prev_action_probs = np.zeros([len(ALL_ACTION_LIST)])
                prev_action_probs[prev_action_idx] = 1
            
            item_action_probs.append(cur_action_probs)
            item_prev_action_probs.append(prev_action_probs)
            item_actions.append(cur_action_idx)
        
        demon_item = {
            'targets': np.array(item_targets),
            'images': np.array(item_images),
            'features': np.array(item_features),
            'action_probs': np.array(item_action_probs),
            'prev_action_probs': np.array(item_prev_action_probs),
            'actions': np.array(item_actions)
        }
        expert_demonstration.append(demon_item)
        # print(f"finish {idm} demon, {i_demon} action")
    
    print(f"num all demonstration:{len(expert_demonstration)}")
    
    with open('./data_split/expert_demonstration_TF.pkl', 'wb') as f:
        pickle.dump(expert_demonstration, f)


if __name__ == '__main__':
    # 100 training tasks
    # generate_split_data("train", 20, 100, 0, 30, 0)
    
    # 20 val tasks
    # generate_split_data("val", 60, 12, 8, 4, 3)
    
    # 50 test tasks
    # generate_split_data("test", 50, 30, 20, 10, 7)
    
    # generate_metadata_expert_demonstration_IL()
    get_expert_demonstration_TF()