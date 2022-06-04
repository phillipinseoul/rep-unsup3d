'''
run train/test/demo
'''
import argparse
import yaml
from unsup3d.__init__ import *
from unsup3d.train import Trainer
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def run(args):
    # load yaml file
    with open(args.configs) as f:
        configs = yaml.safe_load(f)

    #init_settings(configs)
    # load trainer
    trainer = Trainer(configs)
    
    if configs['run_train']:
        print('start training!')
        trainer.train()
    else:
        print('start testing!')
        trainer.test()
    print('run complete!')


'''TODO: run train/test'''
if __name__ == "__main__":
    # set configurations
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--configs', 
        help='Set configurations file path', 
        type=str, 
        default='configs/ablation/config_template.yaml'
    )
    
    # parser.add_argument('--num_workers', help='Set number of workers', type=int, default=4)
    # parser.add_argument('--use_gpu', help='Set the usage of GPU', type=bool, default=True)
    '''TODO: add more arguments'''
    args = parser.parse_args()
    
    run(args)