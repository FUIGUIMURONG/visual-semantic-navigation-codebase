from ai2thor.controller import Controller
import matplotlib.pyplot as plt
import os
import json
import cv2
import torch
import torchvision.transforms as transforms
import h5py
import numpy as np
import sys

sys.path.append("..")
from models.resnet import resnet18

Action_List = ["MoveAhead", "MoveRight", "MoveLeft", "RotateRight", "RotateLeft"]
rotations = [0, 90, 180, 270]


def save_single_scene_image(scene_name):
    controller = Controller(
        gridSize = 0.25,
        scene = scene_name 
    )
    reachable_pos = controller.step(action = 'GetReachablePositions').metadata['actionReturn']
    print(f'scene {scene_name}: {len(reachable_pos)} reachable positions')

    sav_dir = f'./scene_data/{scene_name}/'
    if os.path.exists(sav_dir) == False:
        os.makedirs(sav_dir, exist_ok = True)
    f1 = h5py.File(os.path.join(sav_dir, 'images.h5'), 'w')

    for pos in reachable_pos:
        x = pos['x']
        y = pos['y']
        z = pos['z']
        for rot in rotations:
            event = controller.step(
                action = 'Teleport', 
                position = dict(x = x, y = y, z = z), 
                rotation = dict(x = 0.0, y = rot, z = 0.0), 
                horizon = 0,
                standing = True
            )
            image_sav_path = os.path.join(sav_dir, '{:.2f}_{:.2f}_{:d}.jpg'.format(x, z, rot))
            plt.imsave(image_sav_path, event.frame)
            key = "{:0.2f}|{:0.2f}|{:d}|0".format(x, z, int(rot))
            f1[key] = event.frame
    f1.close()
    print(f'scene {scene_name} finished.')


def save_scenes_images(scene_list):
    for scene in scene_list:
        save_single_scene_image(scene)


def get_next_state(action, x, z, rotation, grid_size = 0.25):
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
    return (x, z, rotation)


def generate_single_graph(scene_name):
    controller = Controller(
        gridSize = 0.25,
        scene = scene_name 
    )
    reachable_pos = controller.step(action = 'GetReachablePositions').metadata['actionReturn']
    print(f'scene {scene_name}: {len(reachable_pos)} reachable positions')
    y = reachable_pos[0]['y']

    states = []
    nodes = []
    links = []
    for point in reachable_pos:
        x = point['x']
        z = point['z']
        for rotation in rotations:
            state = "{:0.2f}|{:0.2f}|{:d}|0".format(x, z, int(rotation))
            nodes.append({'id': state})
            states.append(state)
    
    for state in states:
        x = float(state.split('|')[0])
        z = float(state.split('|')[1])
        rotation = int(state.split('|')[2])
        horizon = int(state.split('|')[3])
        for action in Action_List:
            new_state = get_next_state(action, x, z, rotation, 0.25)
            if new_state != None:
                new_x, new_z, new_rot = new_state
                new_state_str = "{:0.2f}|{:0.2f}|{:d}|0".format(new_x, new_z, int(new_rot))
                if new_state_str in states:
                    links.append({"source": state, "target": new_state_str})
    
    graph_dict = {
        "directed": bool(1), 
        "multigraph": bool(0), 
        "graph": {},
        "nodes": nodes,
        "links": links
    }

    sav_dir = f'./scene_data/{scene_name}/'
    if os.path.exists(sav_dir) == False:
        os.makedirs(sav_dir, exist_ok = True)
    graph_sav_path = os.path.join(sav_dir, "graph.json")
    grid_sav_path = os.path.join(sav_dir, 'grid.json')
    with open(graph_sav_path, 'w') as f:
        f.write(json.dumps(graph_dict, indent = 4))
    grid_dict = {
        'y': y,
        'grid': states
    }
    with open(grid_sav_path, 'w') as f:
        f.write(json.dumps(grid_dict, indent = 4))
    print(f"scene {scene_name} finished.")


def generate_scenes_graph(scene_list):
    for scene in scene_list:
        generate_single_graph(scene)


def preprocess(frame, output_size = (224, 224)):
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # new_h, new_w = output_size
    # new_frame = cv2.resize(frame, (new_h, new_w))
    # new_frame = new_frame.transpose((2, 0, 1))

    # for channel, _ in enumerate(new_frame):
    #     new_frame[channel] = new_frame[channel] / 255.0
    #     new_frame[channel] = new_frame[channel] - mean[channel]
    #     new_frame[channel] = new_frame[channel] * 1.0 / std[channel]
    
    # new_frame = torch.FloatTensor(new_frame)
    # new_frame = new_frame.cuda()
    # new_frame = new_frame.unsqueeze(0)

    images = torch.from_numpy(frame.copy()).float()
    images = images.permute(2, 0, 1)
    images = transforms.Resize((224, 224))(images)
    images = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(images).unsqueeze(0)

    return images


def get_single_resnet18_feature(scene_name):
    controller = Controller(
        gridSize = 0.25,
        scene = scene_name
    )
    reachable_pos = controller.step(action = 'GetReachablePositions').metadata['actionReturn']
    print(f'scene {scene_name}: {len(reachable_pos)} reachable positions')

    resnet = resnet18().cuda()
    sav_dir = f'./scene_data/{scene_name}/'
    if os.path.exists(sav_dir) == False:
        os.makedirs(sav_dir, exist_ok = True)
    f1 = h5py.File(os.path.join(sav_dir, 'resnet18_features.h5'), 'w')
    
    for point in reachable_pos:
        x = point['x']
        y = point['y']
        z = point['z']
        for rot in rotations:
            event = controller.step(
                action = 'Teleport', 
                position = dict(x = x, y = y, z = z), 
                rotation = dict(x = 0.0, y = rot, z = 0.0), 
                horizon = 0,
                standing = True
            )
            frame = preprocess(event.frame).cuda()
            feature = resnet(frame).cpu().detach().numpy()  # 1 x 512 x 1 x 1
            key = "{:0.2f}|{:0.2f}|{:d}|0".format(x, z, int(rot))
            f1[key] = feature
    f1.close()
    print(f'scene {scene_name} finished.')


def get_scenes_resnet18_feature(scene_list):
    for scene in scene_list:
        get_single_resnet18_feature(scene)


def generate_single_visible_object(scene_name):
    controller = Controller(
        gridSize = 0.25,
        visibilityDistance = 1.5,
        scene = scene_name
    )
    reachable_pos = controller.step(action = 'GetReachablePositions').metadata['actionReturn']
    print(f'scene {scene_name}: {len(reachable_pos)} reachable positions')

    sav_dir = f'./scene_data/{scene_name}/'
    if os.path.exists(sav_dir) == False:
        os.makedirs(sav_dir, exist_ok = True)
    
    visible_objects = {}
    objects_where_visible = {}

    for point in reachable_pos:
        x = point['x']
        y = point['y']
        z = point['z']
        for rot in rotations:
            event = controller.step(
                action = 'Teleport', 
                position = dict(x = x, y = y, z = z), 
                rotation = dict(x = 0.0, y = rot, z = 0.0), 
                horizon = 0,
                standing = True
            )
            obj_metadata = event.metadata["objects"]
            cur_visible_objects = [obj["objectId"] for obj in obj_metadata if obj["visible"] == True]
            pos_key = "{:0.2f}|{:0.2f}|{:d}|0".format(x, z, int(rot))
            visible_objects[pos_key] = cur_visible_objects
            for objId in cur_visible_objects:
                if objId not in objects_where_visible.keys():
                    objects_where_visible[objId] = []
                objects_where_visible[objId].append(pos_key)
    
    visobj_sav_path = os.path.join(sav_dir, 'visible_objects.json')
    objvis_sav_path = os.path.join(sav_dir, 'objects_where_visible.json')
    with open(visobj_sav_path, 'w') as f:
        f.write(json.dumps(visible_objects, indent = 4))
    with open(objvis_sav_path, 'w') as f:
        f.write(json.dumps(objects_where_visible, indent = 4))
    print(f'scene {scene_name} finished.')


def generate_scenes_visible_object(scene_list):
    for scene in scene_list:
        generate_single_visible_object(scene)


if __name__ == "__main__":
    scene_list = [f'FloorPlan{idx}' for idx in range(221, 230)]
    # save_scenes_images(scene_list)
    # generate_scenes_graph(scene_list)
    get_scenes_resnet18_feature(scene_list)
    # generate_scenes_visible_object(scene_list)