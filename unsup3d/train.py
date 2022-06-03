'''
Train code for PhotoGeoAE
'''
import pstats
import torch
import torch.optim as optims
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import os.path as path
import os
import random
import numpy as np

from unsup3d.__init__ import *
from unsup3d.model import PhotoGeoAE
from unsup3d.dataloader import CelebA, BFM

# initial configurations 
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.autograd.set_detect_anomaly(False)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.cuda.manual_seed_all(random_seed)


class Trainer():
    def __init__(self, configs, model = None): # model is for debug(05/09)
        '''initialize params (to be implemented)'''
        self.max_epoch = configs['num_epochs']
        self.img_size = configs['img_size']
        self.batch_size = configs['batch_size']
        self.learning_rate = configs['learning_rate']
        self.is_train = configs['run_train']
        self.load_chk = configs['load_chk']

        if self.load_chk:
            self.load_path = configs['load_path']
        else:
            self.load_path = None

        self.epoch = 0
        self.step = 0
        self.best_loss = 1e10
        self.configs = configs

        '''path relevant'''
        now = time.localtime()
        curr_time = "_%02d_%02d__%02d_%02d"%(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
        self.exp_name = configs['exp_name'] + curr_time
        self.exp_path = path.join(configs['exp_path'], self.exp_name)
        os.makedirs(self.exp_path, exist_ok=True)
        self.save_path = path.join(self.exp_path, 'models')
        os.makedirs(self.save_path, exist_ok=True)
        self.best_path = path.join(self.save_path, 'best.pt')

        '''logger setting'''
        self.writer = SummaryWriter(path.join(self.exp_path, 'logs'))
        self.save_epoch = configs['save_epoch']
        self.fig_step = configs['fig_plot_step']
        print(f'logs stored at {self.exp_path}')

        '''implement dataloader'''
        if configs['dataset'] == "celeba":
            self.datasets = CelebA(setting = 'train')
            self.val_datasets = CelebA(setting = 'val')
        elif configs['dataset'] == "bfm":
            self.datasets = BFM(setting = 'train')
            self.val_datasets = BFM(setting = 'val')

        self.dataloader = DataLoader(
            self.datasets,
            batch_size= self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,             
        )

        if self.val_datasets is not None:
            self.val_dataloader = DataLoader(
                self.val_datasets,
                batch_size= self.batch_size,
                shuffle=False,
                num_workers=8,
                drop_last=True,         
            )
        
        '''select GPU'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        '''define model'''
        if is_debug:
            self.model = model.to(self.device)                                 
        else:
            self.model = PhotoGeoAE(configs).to(self.device)
        self.model.set_logger(self.writer)

        '''define optimizer and scheduler'''
        self.optimizer = optims.Adam(
            # params = self.model.parameters(),
            params = self.model.imgDecomp.parameters(),
            lr = self.learning_rate,
            betas=(0.9, 0.999), 
            weight_decay=5e-4       # from author's code setting (05/22 inhee)
        )

        self.scheduler = optims.lr_scheduler.LambdaLR(
            optimizer = self.optimizer,
            lr_lambda = lambda epoch: 0.95 ** epoch,
        )

        '''load_model and optimizer state'''
        if self.load_chk:
            self.load_model(self.load_path if self.load_path is not None else self.best_path)
        
    def train(self):
        init_epch = self.epoch
        for epch in range(init_epch, self.max_epoch):
            epch_loss = self._train()       # train a single epoch
            self.epoch = epch

            if epch_loss < self.best_loss:
                # save best model
                self.save_model(epch_loss)      
                self.best_loss = epch_loss

            '''
            if self.epoch % self.save_epoch == 0 or self.epoch == (self.max_epoch - 1):
                # save periodically
                self.save_model(epch_loss)
            '''
            self.writer.add_scalar("loss_epch/train", epch_loss, self.epoch)

    def _train(self):
        '''train model (single epoch)'''
        epch_loss = 0
        cnt = 0
        for i, inputs in tqdm(enumerate(self.dataloader, 0)):
            if self.model.use_gt_depth:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)
            
            self.optimizer.zero_grad()

            if test_supervised:
                loss = self.model(inputs)
            else:
                losses = self.model(inputs)
                loss = torch.mean(losses)

            loss.backward()

            # add gradient clipping (06/01)
            torch.nn.utils.clip_grad_norm_(self.model.imgDecomp.parameters(), max_norm=5)
            self.optimizer.step()

            # calculate epch_loss
            epch_loss += loss.detach().cpu()
            self.writer.add_scalar("Loss_step/train", loss.detach().cpu().item(), self.step)
            self.model.loss_plot(self.step)

            if self.step % self.fig_step == 0:
                self.model.visualize(self.step)

            cnt += 1
            self.step += 1
        
        if use_sched:
            self.scheduler.step()
        return epch_loss/cnt

    def load_model(self, PATH):
        '''
        save loaded model (05/08)
        '''
        chkpt = torch.load(PATH)
        self.model.load_state_dict(chkpt['model_state_dict'])
        self.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        self.epoch = chkpt['epoch']
        self.step = chkpt['step']
        self.best_loss = chkpt['loss']

    def save_model(self, loss):
        if loss < self.best_loss:
            PATH = path.join(self.save_path, 'best.pt')
        else:
            # saving chkpts periodically
            PATH = path.join(self.save_path, 'epoch_'+str(self.epoch)+'.pt')
        
        torch.save({
            'epoch': self.epoch,
            'step' : self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, PATH)

        print("mode saved as ", PATH)

    def _val(self):
        '''validate model and plot testing images'''
        self.model.eval()

        with torch.no_grad():
            for i, inputs in tqdm(enumerate(self.val_dataloader, 0)):
                inputs = inputs.to(self.device)
                losses = self.model(inputs)

    def _test(self):
        '''test model'''
        pass

