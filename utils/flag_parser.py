import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    

def parse_arguments():
    parser = argparse.ArgumentParser(description = "VSN")
    parser.add_argument(
        "--train_or_test", 
        type = str,
        default = "train", 
        help = "train or val or test."
    )

    parser.add_argument(
        "--test_setting",
        type = str,
        default = "all", 
        help = "test testing, can be all or seen or unseen."
    )

    parser.add_argument(
        "--algorithm",
        type = str,
        default = "IL", 
        help = "The training algorithm, can be IL or RL."
    )

    parser.add_argument(
        "--tb_dir", 
        type = str,
        default = "runs/", 
        help = "folder to save logs and models."
    )

    parser.add_argument(
        "--exp_data_dir",
        type = str,
        default = "./dataset/data_split/",
        help = "The path of the training / validation / testing data."
    )

    parser.add_argument(
        "--IL_model",
        type = str,
        default = "BaseModel",
        help = "The model type for IL, can be BaseModel or GCN."
    )

    parser.add_argument(
        "--IL_lr",
        type = float,
        default = 0.0005,
        help = "The learning rate of IL."
    )

    parser.add_argument(
        "--IL_batch_size",
        type = int,
        default = 64,
        help = "The batch size of IL."
    )

    parser.add_argument(
        "--num_workers",
        type = int,
        default = 4,
        help = "The number of used workers when training (default: 4)."
    )

    parser.add_argument(
        "--pin_memory",
        type = str2bool,
        default = False,
        help = "If pin_memory is set to True (default: False)."
    )

    parser.add_argument(
        "--IL_epochs", 
        type = int,
        default = 2000,
        help = "The number of IL training epochs."
    )

    parser.add_argument(
        "--IL_save_epochs", 
        type = int,
        default = 500,
        help = "The number of saving epochs for IL model when training."
    )

    parser.add_argument(
        "--ResNet18_path", 
        type = str,
        default = "./models/resnet18.pth",
        help = "The path of the resnet18 parameters file."
    )

    parser.add_argument(
        "--test_IL_load_model",
        type = str,
        default = "runs/load_models/IL.pth", 
        help = "The path of the loaded trained model when testing with IL."
    )

    parser.add_argument(
        "--test_RL_load_model",
        type = str,
        default = "runs/load_models/BaseModel.pth", 
        help = "The path of the loaded trained model when testing with IL."
    )

    parser.add_argument(
        "--features_or_images", 
        type = str2bool, 
        default = True, 
        help = "The input is the features of the images. (default: resnet18 features)."
    )

    parser.add_argument(
        "--n_objects", 
        type = int, 
        default = 83, 
        help = "The number of objects in the scene."
    )

    parser.add_argument(
        "--adjmat_dir", 
        type = str, 
        default = "./dataset/adjmat.dat", 
        help = "The path of the adjacency matrix."
    )

    parser.add_argument(
        "--obj_file_dir", 
        type = str, 
        default = "./dataset/objects.txt", 
        help = "The path of the object file."
    )
    
    parser.add_argument(
        "--glove_dim",
        type = int,
        default = 300,
        help = "The number of dimensions of the glove vector to use."
    )

    parser.add_argument(
        "--glove_embed_dim",
        type = int,
        default = 256,
        help = "The number of dimensions of processed glove embedding"
    )
    
    parser.add_argument(
        "--action_space", 
        type = int, 
        default = 6, 
        help = "The number of actions in the action space."
    )

    parser.add_argument(
        "--action_dim", 
        type = int, 
        default = 32, 
        help = "The number of dimensions of action embedding."
    )

    parser.add_argument(
        "--hidden_state_dim", 
        type = int, 
        default = 512, 
        help = "The number of dimensions of hidden state of LSTM."
    )

    parser.add_argument(
        "--gcn_dim", 
        type = int, 
        default = 512, 
        help = "The number of dimensions of gcn features."
    )

    parser.add_argument(
        "--offline_data_dir",
        type = str,
        default = "./dataset/scene_data/",
        help = "where dataset is stored(grid.json, graph.json).",
    )

    parser.add_argument(
        "--grid_file_name",
        type = str,
        default = "grid.json",
        help = "The file name of the reasonable grid file.",
    )

    parser.add_argument(
        "--graph_file_name",
        type = str,
        default = "graph.json",
        help = "The file name of the graph file.",
    )

    parser.add_argument(
        "--input_file_name",
        type = str,
        default = "resnet18_features.h5",
        help = "The file name of the input file, can be resnet18_features.h5 or images.h5",
    )

    parser.add_argument(
        "--object_metadata_file_name",
        type = str,
        default = "objects_where_visible.json",
        help = "The file name of the object metadata file, can be objects_where_visible.json or visible_objects.json",
    )

    parser.add_argument(
        "--glove_dir",
        type = str,
        default = "./dataset",
        help = "where the glove files are stored.",
    )

    parser.add_argument(
        "--RL_model",
        type = str,
        default = "BaseModel",
        help = "The model type for IL, can be BaseModel or GCN."
    )

    parser.add_argument(
        "--RL_lr",
        type = float,
        default = 0.0001,
        help = "learning rate for RL (default: 0.0001)."
    )

    parser.add_argument(
        "--amsgrad",
        type = str2bool, 
        default = True, 
        help = "Adam optimizer amsgrad parameter."
    )

    parser.add_argument(
        "--shared",
        type = str2bool, 
        default = True, 
        help = "True - shared optimizer, False - No shared optimizer."
    )

    parser.add_argument(
        "--gpu_ids",
        type = int,
        default = [0],
        nargs = "+",
        help = "GPUs to use [-1 CPU only] (default: -1). Notice that -1 can not be input together with other GPU ids."
    )

    parser.add_argument(
        "--seed", 
        type = int, 
        default = 123, 
        help = "random seed."
    )

    parser.add_argument(
        "--num_steps", 
        type = int, 
        default = 50, 
        help = "number of forward steps in A3C."
    )

    parser.add_argument(
        "--If_IL_pretrain", 
        type = str2bool, 
        default = False, 
        help = "If use the trained IL model to initize the model for RL training."
    )

    parser.add_argument(
        "--load_IL_path", 
        type = str, 
        default = "runs/load_models/IL.pth", 
        help = "The path of the loaded IL model for RL training."
    )

    parser.add_argument(
        "--max_RL_episode", 
        type = int, 
        default = 1000000, 
        help = "The maximum number of episodes for RL training."
    )

    parser.add_argument(
        "--n_record_RL", 
        type = int, 
        default = 100, 
        help = "How often to record training information, the number of record eposodes."
    )

    parser.add_argument(
        "--RL_save_episodes", 
        type = int,
        default = 100000,
        help = "The number of saving episodes for RL model when training."
    )

    parser.add_argument(
        "--gamma",
        type = float,
        default = 0.99,
        help = "discount factor for rewards (default: 0.99)",
    )

    parser.add_argument(
        "--tau",
        type = float,
        default = 1.00,
        help = "parameter for GAE (default: 1.00)",
    )

    parser.add_argument(
        "--beta", 
        type = float, 
        default = 1e-2, 
        help = "entropy regularization term"
    )

    parser.add_argument(
        "--max_episode_length",
        type = int,
        default = 100,
        help = "maximum length of an episode (default: 100)",
    )

    parser.add_argument(
        "--dropout_rate",
        type = float,
        default = 0.25,
        help = "The dropout ratio to use.",
    )

    parser.add_argument(
        "--test_out_dir",
        type = str,
        default = "res/",
        help = "The output directory of the test result file.",
    )

    parser.add_argument(
        "--strict_done",
        type = str2bool,
        default = False,
        help = "If use the strict success judgement rules to judge if the task is successful",
    )


    args = parser.parse_args()
    args.glove_file = "{}/glove_map{}d.hdf5".format(args.glove_dir, args.glove_dim)

    return args
