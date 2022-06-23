from dataloaders.dense_to_sparse import UniformSampling, ORBSampling
from dataloaders.nyu_dataloader import NYUDataset
import os
import numpy as np
import torch

import yaml
from easydict import EasyDict as edict

package_directory = os.path.dirname(os.path.abspath(__file__))

def create_data_loaders(args):

    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == ORBSampling.name:
        sparsifier = ORBSampling(num_samples=args.num_samples, max_depth=max_depth)

    train_dataset = NYUDataset(traindir, type='train',
        modality="rgbd", sparsifier=sparsifier)
    val_dataset = NYUDataset(valdir, type='val',
        modality="rgbd", sparsifier=sparsifier)

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None,
        worker_init_fn=lambda work_id:np.random.seed(work_id))
        # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

def load_config(path):
    filename = os.path.join(package_directory, path)
    with open(filename, 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    return edict(config_data)

def save_state(config, model):
    print('==> Saving model ...')
    env_name =  + 'model' + str(config.manual_seed)
    save_path = os.path.join('checkpoints', env_name)
    os.makedirs(save_path, exist_ok=True)
    model_state_dict = model.state_dict()
    state_dict = {
        'model': model_state_dict,
    }
    torch.save(state_dict, os.path.join(save_path, 'result.pth'))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_rmse = 0
        self.sum_mae = 0 
        self.sum_huber = 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_huber += n*result.huber
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time

    def get_avg(self):
        return self.sum_huber / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, self.sum_data_time / self.count, self.sum_gpu_time / self.count


class Result:
    def __init__(self, pred, scale_factor):
        pred = pred.cpu().numpy()
        scale_factor = scale_factor.cpu().numpy()

        self.mae = self.mae(pred, scale_factor)
        self.rmse = self.rmse(pred, scale_factor)
        self.huber = 0

    def rmse(self,pred, scale_factor):
        return np.sqrt(((pred - scale_factor)**2).mean())
    
    def mae(self, pred, scale_factor):
        return np.absolute(pred -scale_factor).mean()


