import torch
import torch.nn as nn
import json
import os
import time
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import tqdm
import pickle
import numpy as np
from tensorboardX import SummaryWriter

from utils.flag_parser import parse_arguments
from utils.utils import reset_player, compute_single_spl_offline
from models import BaseModel, GCN
from controller_offline import OfflineController
from agent import NavAgent


class IL_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(IL_CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction = 'none')

    def forward(self, dist, target, mask):
        bs, seq_len = dist.shape[0], dist.shape[1]
        dist = dist.view(bs*seq_len, -1)
        target = target.view(-1)
        mask = mask.view(-1)
        loss = self.loss(dist, target)
        loss = (loss * mask).sum() / mask.sum()
        return loss


class Dataset(Dataset):
    def __init__(self, args, split = "train", setting = "seen"):
        super(Dataset, self).__init__()
        assert (split == "train") or (split == "val") or (split == "test"), "Invalid Split input."
        assert (setting == "seen") or (setting == "unseen") or (setting == "all"), "Invalid Setting input."
        if split == "train":
            assert setting == "seen", "Invalid setting for training dataset"
        
        self.args = args
        self.split = split
        self.metadata = []
        self.data = []

        data_split_name = f"{split}_{setting}.json"
        with open(os.path.join(args.exp_data_dir, data_split_name), 'r') as f:
            self.metadata = json.load(f)
        
        if split == "train":
            with open(os.path.join(args.exp_data_dir, "expert_demonstration_TF.pkl"), "rb") as f:
                all_data = pickle.load(f)
            
            for idx, item in enumerate(all_data):
                targets = torch.FloatTensor(item['targets'])
                
                final_images = torch.zeros((len(item['actions']), 3, 224, 224))
                for idm, image in enumerate(item['images']):
                    item_image = torch.from_numpy(image.copy()).float()
                    item_image = item_image.permute((2, 0, 1))
                    item_image = transforms.Resize((224, 224))(item_image)
                    item_image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(item_image)
                    final_images[idm] = item_image

                features = torch.FloatTensor(item['features'])
                action_probs = torch.FloatTensor(item['action_probs'])
                prev_action_probs = torch.FloatTensor(item['prev_action_probs'])
                actions = torch.LongTensor(item['actions'])

                if idx % 100 == 99:
                    print(f"load data {idx + 1}")

                self.data.append(
                    {
                        'targets': targets, 
                        'images': final_images, 
                        'features': features, 
                        'action_probs': action_probs,
                        'prev_action_probs': prev_action_probs, 
                        'actions': actions
                    }
                )
        
        else:
            self.data = self.metadata
        print('load data finished.')


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        return item
    

    def collate_fn(self, batch_samples):
        batch_size = len(batch_samples)
        length = max([len(item['actions']) for item in batch_samples])
        
        targets = torch.zeros((batch_size, length, batch_samples[0]['targets'].shape[1]))
        masks = torch.zeros((batch_size, length))
        
        images_shape = [batch_size, length]
        single_image_shape = [shape for shape in batch_samples[0]['images'].shape[1:]]
        images_shape = images_shape + single_image_shape
        images = torch.zeros(images_shape)

        features_shape = [batch_size, length]
        single_feature_shape = [shape for shape in batch_samples[0]['features'].shape[1:]]
        features_shape = features_shape + single_feature_shape
        features = torch.zeros(features_shape)

        action_probs = torch.zeros((batch_size, length, batch_samples[0]['action_probs'].shape[1]))
        prev_action_probs = torch.zeros((batch_size, length, batch_samples[0]['prev_action_probs'].shape[1]))
        actions = torch.zeros((batch_size, length), dtype = torch.long)

        for idx, item in enumerate(batch_samples):
            targets[idx, :len(item['actions']), :] = item['targets']
            images[idx, :len(item['actions']), :, :, :] = item['images']
            features[idx, :len(item['actions']), :, :, :] = item['features']
            action_probs[idx, :len(item['actions']), :] = item['action_probs']
            prev_action_probs[idx, :len(item['actions']), :] = item['prev_action_probs']
            actions[idx, :len(item['actions'])] = item['actions']
            masks[idx, :len(item['actions'])] = 1
        
        return {'targets': targets, 
                'images': images, 
                'features': features, 
                'action_probs': action_probs,
                'prev_action_probs': prev_action_probs, 
                'actions': actions,
                'masks': masks
            }


def train_IL(args, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    full_time = time.localtime(time.time())
    year, mon, day, hour, min_ = full_time.tm_year, full_time.tm_mon, full_time.tm_mday, full_time.tm_hour, full_time.tm_min
    tb_dir = os.path.join(args.tb_dir, 'IL_{}_{}_{}_{}_{}_{}'.format(year, mon, day, hour, min_, args.IL_model))
    if os.path.exists(tb_dir) == False:
        os.makedirs(tb_dir, exist_ok = True)

    args_str = {}
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
        args_str[k] = args.__dict__[k]
    with open(os.path.join(tb_dir, 'args_config.json'), 'w') as f:
        f.write(json.dumps(args_str, indent = 4))
    
    model = None
    assert (args.IL_model == 'BaseModel' or args.IL_model == 'GCN')
    if args.IL_model == 'BaseModel':
        model = BaseModel(args)
    elif args.IL_model == 'GCN':
        model = GCN(args)
    
    if -1 in args.gpu_ids:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu_ids[0]))
    model.to(device)

    optimizer = torch.optim.AdamW([param for name, param in model.named_parameters() if param.requires_grad == True], lr = args.IL_lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.1)

    train_dataset = Dataset(args, 'train')
    train_dataloader = DataLoader(train_dataset, args.IL_batch_size, shuffle = True, collate_fn = train_dataset.collate_fn, num_workers = args.num_workers, pin_memory = args.pin_memory)
    
    loss_func = IL_CrossEntropyLoss()
    writer = SummaryWriter(log_dir = os.path.join(tb_dir, 'IL_train'))
    if args.pin_memory and args.num_workers > 1:
        args.non_blocking = True
    else:
        args.non_blocking = False

    total_iter = 0
    for epoch in range(args.IL_epochs):
        mean_loss = 0.
        mean_num = 0
        for idx, data in enumerate(train_dataloader):

            target = data['targets'].to(device, non_blocking = args.non_blocking)
            if args.features_or_images == True:
                inputs = data['features'].to(device, non_blocking = args.non_blocking)
            else:
                inputs = data['images'].to(device, non_blocking = args.non_blocking)
            prev_action_probs = data['prev_action_probs'].to(device, non_blocking = args.non_blocking)
            actions = data['actions'].to(device, non_blocking = args.non_blocking)
            masks = data['masks'].to(device, non_blocking = args.non_blocking)

            actor_out, critic_out, (hs, cs) = model(target, inputs, prev_action_probs)
            # bs, seq_len = actor_out.shape[0], actor_out.shape[1]

            actor_loss = loss_func(actor_out, actions, masks)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 20)
            actor_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mean_loss += actor_loss.item()
            mean_num += 1

            if idx % 5 == 4:
                writer.add_scalar('train/actor_loss', actor_loss.item(), total_iter)
                writer.add_scalar('train/IL_lr', optimizer.param_groups[0]['lr'], total_iter)
                print('[epoch {} batch {}] loss: {:.5f}'.format(epoch + 1, idx + 1, mean_loss / mean_num))
                mean_loss = 0
                mean_num = 0
            total_iter += 1

        if epoch % 2000 == 1999:
            for params in optimizer.param_groups:       
                params['lr'] *= 0.5

        if epoch % args.IL_save_epochs == args.IL_save_epochs - 1:
            torch.save(model.state_dict(), os.path.join(tb_dir, 'epoch_{}.pth'.format(epoch + 1)))
        # scheduler.step()


def test_IL(args, split, setting):
    assert (split == 'val' or split == 'test')
    assert (setting == 'seen' or setting == 'unseen' or setting == 'all')
    with open(os.path.join(args.exp_data_dir, f"{split}_{setting}.json"), "r") as f:
        test_data = json.load(f)

    if -1 in args.gpu_ids:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpu_ids[0]}")

    model = None
    assert (args.IL_model == 'BaseModel' or args.IL_model == 'GCN')
    if args.IL_model == 'BaseModel':
        model = BaseModel(args)
    elif args.IL_model == 'GCN':
        model = GCN(args)
    state_dict = torch.load(args.test_IL_load_model)
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
        test_spl.append(single_spl)
        test_path_length.append(path_length)
        print(f"{i_task} episode finished.")
    
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
    res_sav_path = os.path.join(args.test_out_dir, f"res_IL_{args.IL_model}_{split}_{setting}.json")
    with open(res_sav_path, "w") as f:
        f.write(json.dumps(res_dict, indent = 4))



if __name__ == "__main__":
    args = parse_arguments()
    # train_IL(args)
    test_IL(args, "test", "seen")
    # test_IL(args, "test", "unseen")
    # test_IL(args, "test", "all")
