""" Offline Controller. """

import importlib
import json
import copy
import time
import random
import os
import platform
import numpy as np
import h5py
from scipy import spatial
from ai2thor.controller import Controller
from dataset.constants import ACTION_LIST


class ThorAgentState:
    """ Representation of the state in the grid level which includes the position, horizon and rotation. """

    def __init__(self, x, y, z, rotation, horizon):
        self.x = round(x, 2)
        self.y = y
        self.z = round(z, 2)
        self.rotation = round(rotation)
        self.horizon = round(horizon)

    @classmethod
    def get_state_from_event(cls, event, forced_y = None):
        """ Extracts a state from an event. """
        state = cls(
            x = event.metadata["agent"]["position"]["x"],
            y = event.metadata["agent"]["position"]["y"],
            z = event.metadata["agent"]["position"]["z"],
            rotation = event.metadata["agent"]["rotation"]["y"],
            horizon = event.metadata["agent"]["cameraHorizon"],
        )
        if forced_y != None:
            state.y = forced_y
        return state

    def __eq__(self, other):
        """ If we check for exact equality then we get issues.
            For now we consider this 'close enough'. """
        if isinstance(other, ThorAgentState):
            return (
                self.x == other.x
                and
                # self.y == other.y and
                self.z == other.z
                and self.rotation == other.rotation
                and self.horizon == other.horizon
            )
        return NotImplemented

    def __str__(self):
        """ Get the string representation of a state. """
        """
        return '{:0.2f}|{:0.2f}|{:0.2f}|{:d}|{:d}'.format(
            self.x,
            self.y,
            self.z,
            round(self.rotation),
            round(self.horizon)
        )
        """
        return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
            self.x, self.z, round(self.rotation), round(self.horizon)
        )

    def position(self):
        return dict(x = self.x, y = self.y, z = self.z)


class OfflineControllerEvent:
    """ 
    A stripped down version of an event. Only contains lastActionSuccess, sceneName,
    and optionally state and frame. Does not contain the rest of the metadata. 
    """

    def __init__(self, last_action_success, scene_name, state = None, frame = None):
        self.metadata = {
            "lastActionSuccess": last_action_success,
            "sceneName": scene_name,
        }
        if state is not None:
            self.metadata["agent"] = {}
            self.metadata["agent"]["position"] = state.position()
            self.metadata["agent"]["rotation"] = {
                "x": 0.0,
                "y": state.rotation,
                "z": 0.0,
            }
            self.metadata["agent"]["cameraHorizon"] = state.horizon
        self.frame = frame


class OfflineController:
    """
    A stripped down version of the offline controller. Only allows for a few given actions.
    
    Data is stored in offline_data_dir/<scene_name>/.
    grid.json records the reachable positions in the specific scene.
    graph.json records the relationships between reachable positions in the specific scene.
    
    input_file_name indicates the name of the input data file. 
    It can be the raw images 'images.h5' or the visual features extracted by ResNet18 'resnet18_features.h5'.

    object_metadata_file_name indicates the name of the metadata file of objects.
    It can be 'visible_objects.json', which records the visible objects in each reachable positions.
    It also can be 'objects_where_visible.json', which records the visible positions for each object.
    """

    def __init__(
        self,
        args,
        offline_data_dir = "./dataset/scene_data/",
        grid_file_name = "grid.json",
        graph_file_name = "graph.json",
        input_file_name = "resnet18_features.h5",
        object_metadata_file_name = "objects_where_visible.json",
        scene = 'FloorPlan221',
        grid_size = 0.25,
        fov = 90,
        h = 300,
        w = 300,
        actions = ACTION_LIST,
        rotations = [0, 90, 180, 270],
        horizons = [0]
    ):

        super(OfflineController, self).__init__()
        self.args = args
        self.scene_name = scene
        self.grid_size = grid_size
        self.offline_data_dir = offline_data_dir
        self.grid_file_name = grid_file_name
        self.graph_file_name = graph_file_name
        self.input_file_name = input_file_name
        self.object_metadata_file_name = object_metadata_file_name

        self.grid = None
        self.graph = None
        self.object_metadata = None
        self.inputs = None
        self.using_raw_metadata = False
        self.actions = actions
        self.fov = fov
        self.rotations = rotations
        self.horizons = horizons
        self.h = h
        self.w = w

        self.last_event = None
        self.state = None
        self.last_action_success = True

        self.h5py = importlib.import_module("h5py")
        self.nx = importlib.import_module("networkx")
        self.json_graph_loader = importlib.import_module("networkx.readwrite")

        self.reset(self.scene_name)


    def get_full_state(self, x, y, z, rotation = 0.0, horizon = 0.0):
        return ThorAgentState(x, y, z, rotation, horizon)

    def get_state_from_str(self, x, z, rotation = 0.0, horizon = 0.0):
        return ThorAgentState(x, self.y, z, rotation, horizon)


    def reset(self, scene_name = None):
        if scene_name is None:
            scene_name = "FloorPlan221"
            print('Do not provide the reset scene. Reset to FloorPlan221.')

        self.scene_name = scene_name
        scene_data_dir = os.path.join(self.offline_data_dir, self.scene_name)
        with open(os.path.join(scene_data_dir, self.grid_file_name), "r") as f:
            grid_dict = json.load(f)
            self.grid = grid_dict['grid']
            self.y = grid_dict['y']

        with open(os.path.join(scene_data_dir, self.graph_file_name), "r") as f:
            graph_json = json.load(f)
        self.graph = self.json_graph_loader.node_link_graph(graph_json).to_directed()
        
        with open(os.path.join(scene_data_dir, self.object_metadata_file_name), "r") as f:
            self.object_metadata = json.load(f)
            if len(self.object_metadata.keys()) > 0:
                key = list(self.object_metadata.keys())[0]
                try:
                    float(key.split("|")[0])
                    self.using_raw_metadata = True
                except ValueError:
                    self.using_raw_metadata = False

        if self.inputs is not None:
            self.inputs.close()
        self.inputs = self.h5py.File(os.path.join(scene_data_dir, self.input_file_name), "r")

        self.state = self.get_full_state(
            x = float(self.grid[0].split('|')[0]), 
            y = float(self.y), 
            z = float(self.grid[0].split('|')[1]), 
            rotation = float(self.grid[0].split('|')[2])
        )
        self.last_action_success = True
        self.last_event = self._successful_event()
        return self.last_event
    
    def teleport(self, pos):
        self.state = self.get_state_from_str(
            x = float(pos.split("|")[0]),
            z = float(pos.split("|")[1]),
            rotation = float(pos.split("|")[2]),
            horizon = float(pos.split("|")[3])
        )
        self.state.horizon = 0
        self.last_action_success = True
        self.last_event = self._successful_event()
        return self.last_event


    def randomize_state(self):
        random_state = random.choice(list(self.inputs.keys()))
        self.state = self.get_state_from_str(
            x = float(random_state.split("|")[0]),
            z = float(random_state.split("|")[1]),
            rotation = float(random_state.split("|")[2]),
            horizon = float(random_state.split("|")[3])
        )
        self.state.horizon = 0
        self.last_action_success = True
        self.last_event = self._successful_event()
        return self.last_event


    def back_to_start(self, start):
        self.state = start
    

    def get_next_state(self, state, action, deep_copy = True):
        if deep_copy:
            next_state = copy.deepcopy(state)
        else:
            next_state = state
        
        if action == "MoveAhead":
            if state.rotation == 0:
                next_state.z += self.grid_size
            elif state.rotation == 90:
                next_state.x += self.grid_size
            elif state.rotation == 180:
                next_state.z -= self.grid_size
            elif state.rotation == 270:
                next_state.x -= self.grid_size
            else:
                raise Exception("Unknown Rotation")
        elif action == "MoveRight":
            if state.rotation == 0:
                next_state.x += self.grid_size
            elif state.rotation == 90:
                next_state.z -= self.grid_size
            elif state.rotation == 180:
                next_state.x -= self.grid_size
            elif state.rotation == 270:
                next_state.z += self.grid_size
            else:
                raise Exception("Unknown Rotation")
        elif action == "MoveLeft":
            if state.rotation == 0:
                next_state.x -= self.grid_size
            elif state.rotation == 90:
                next_state.z += self.grid_size
            elif state.rotation == 180:
                next_state.x += self.grid_size
            elif state.rotation == 270:
                next_state.z -= self.grid_size
            else:
                raise Exception("Unknown Rotation")
        elif action == "RotateRight":
            next_state.rotation = (state.rotation + 90) % 360
        elif action == "RotateLeft":
            next_state.rotation = (state.rotation - 90 + 360) % 360
        return next_state


    def step(self, action):
        assert (action in self.actions), "Unsupported action."

        next_state = self.get_next_state(self.state, action, True)
        
        if next_state is not None:
            next_state_key = str(next_state)
            neighbors = self.graph.neighbors(str(self.state))
            reachable_state = self.inputs.keys()

            if next_state_key in neighbors and next_state_key in reachable_state:
                self.state = self.get_state_from_str(
                    *[float(x) for x in next_state_key.split("|")]
                )
                self.last_action_success = True
                event = self._successful_event()
                self.last_event = event
                return event

        self.last_action_success = False
        self.last_event.metadata["lastActionSuccess"] = False
        return self.last_event


    def shortest_path(self, source_state, target_state):
        return self.nx.shortest_path(self.graph, str(source_state), str(target_state))


    def optimal_plan(self, source_state, path):
        # self.state = source_state
        current_state = source_state
        actions = []
        i = 1
        while i < len(path):
            for a in self.actions:
                next_state = self.get_next_state(current_state, a, True)
                if str(next_state) == path[i]:
                    actions.append(a)
                    i += 1
                    current_state = next_state
                    break
        return actions


    def shortest_path_to_target(self, source_state, objId, get_plan = False):
        states_where_visible = []
        if self.using_raw_metadata == True:
            for pos in self.object_metadata.keys():
                vis_obj = self.object_metadata[pos]
                if objId in vis_obj:
                    states_where_visible.append(pos)
        else:
            states_where_visible = self.object_metadata[objId]

        states_where_visible = [
            self.get_state_from_str(*[float(x) for x in str_.split("|")])
            for str_ in states_where_visible
        ]

        best_path = None
        best_path_len = 10000
        for t in states_where_visible:
            path = self.shortest_path(source_state, t)
            if len(path) < best_path_len or best_path is None:
                best_path = path
                best_path_len = len(path)
        
        best_plan = None
        if get_plan == True:
            best_plan = self.optimal_plan(source_state, best_path)

        return best_path, best_path_len, best_plan
    

    def shortest_path_to_target_type(self, source_state, obj_type, get_plan = False):
        all_objects = self.all_objects()
        best_path = None
        best_path_len = 10000
        best_plan = None

        for obj in all_objects:
            if obj.split("|")[0] == obj_type:
                tmp_path, tmp_path_len, tmp_plan = self.shortest_path_to_target(source_state, obj, get_plan)
                if tmp_path_len < best_path_len:
                    best_path_len = tmp_path_len
                    best_path = tmp_path
                    best_plan = tmp_plan
        return best_path, best_path_len, best_plan


    def object_is_visible(self, objId):
        if self.using_raw_metadata == True:
            return objId in self.object_metadata[str(self.state)]
        else:
            return str(self.state) in self.object_metadata[objId]


    def _successful_event(self):
        return OfflineControllerEvent(
            self.last_action_success, self.scene_name, self.state, self.get_input()
        )


    def get_input(self):
        return self.inputs[str(self.state)][:]
            

    def all_objects(self):
        if self.using_raw_metadata == True:
            all_objects = []
            for pos in self.object_metadata.keys:
                all_objects += self.object_metadata[pos]
                all_objects = list(set(all_objects))
            return all_objects
        else:
            return self.object_metadata.keys()

