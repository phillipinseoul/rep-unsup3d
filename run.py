'''
run train/test/demo
'''
import argparse
import yaml
from unsup3d import Trainer
from tensorboardX import SummaryWriter

import warnings
#warnings.filterwarnings("error")

def run(args):
    # load yaml file
    with open(args.configs) as f:
        configs = yaml.safe_load(f)

    # load trainer
    trainer = Trainer(configs)
    trainer.train()
    print('run complete!')


'''TODO: run train/test'''
if __name__ == "__main__":
    # set configurations
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configs', 
        help='Set configurations file path', 
        type=str, 
        default='configs/celeba_train_v0.yaml'
    )
    # parser.add_argument('--num_workers', help='Set number of workers', type=int, default=4)
    # parser.add_argument('--use_gpu', help='Set the usage of GPU', type=bool, default=True)
    '''TODO: add more arguments'''
    args = parser.parse_args()

    
    run(args)