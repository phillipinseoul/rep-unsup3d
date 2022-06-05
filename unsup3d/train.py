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
import yaml
import glob
import random
import numpy as np
import zipfile

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

        '''save codes and train settings'''
        code_dir = path.dirname(os.path.realpath(__file__))
        if not path.isdir(code_dir):
            print("failed to dump running codes, dir : ", code_dir)
        else:
            z_f = zipfile.ZipFile(path.join(self.save_path, "dumped_code.zip"), 'w', zipfile.ZIP_DEFLATED)
            flist = []
            flist.extend(glob.glob(path.join(code_dir, '*'+".py"), recursive=True))
            
            for f in flist:
                z_f.write(f)
            z_f.close()


        dump_yaml_name = path.join(self.exp_path, self.exp_name+'.yaml')
        with open(dump_yaml_name, "w") as f:
            yaml.safe_dump(configs, f)



        '''implement dataloader'''
        if configs['dataset'] == "celeba":
            self.datasets = CelebA(setting = 'train')
            self.val_datasets = CelebA(setting = 'val')
            self.test_datasets = CelebA(setting = 'test')
        elif configs['dataset'] == "bfm":
            self.datasets = BFM(setting = 'train')
            self.val_datasets = BFM(setting = 'val')
            self.test_datasets = BFM(setting = 'test')

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

        if self.test_datasets is not None:
            self.test_dataloader = DataLoader(
                self.test_datasets,
                batch_size= self.batch_size,
                shuffle = False,
                num_workers=4,
                drop_last= True
            )
        
        '''select GPU'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        '''define model'''
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

            
            if self.epoch % self.save_epoch == 0 and self.epoch != 0:
                # save periodically
                self.save_model(epch_loss)
            
            self.writer.add_scalar("loss_epch/train", epch_loss, self.epoch)

        print("train finished")
        print("test on test set")
        self.test()

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
            if USE_GRADIENT_CLIP:
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
        
        if USE_SCHED:
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

        print("loaded model from ", PATH)

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

                if i == 0:
                    tot_side_err = self.model.side_error_.view(-1)
                    tot_side_err_v2 = self.model.side_error_v2_.view(-1)
                    tot_mad_err = self.model.mad_error_.view(-1)
                    iter_cnt = 1
                else:
                    iter_cnt += 1
                    tot_mad_err = torch.cat([tot_mad_err, self.model.mad_error_], dim=0)
                    tot_side_err = torch.cat([tot_side_err, self.model.side_error_], dim=0)
                    tot_side_err_v2 = torch.cat([tot_side_err_v2, self.model.side_error_v2_], dim=0)
        
        print("--------------------------------------------------")
        print("side err mean: ", tot_side_err.mean(), "side err std: ", tot_side_err.std())
        print("side err v2 mean: ", tot_side_err_v2.mean(), "side err v2 std: ", tot_side_err_v2.std())
        print("mad err mean: ", tot_mad_err.mean(), "mad err std: ", tot_mad_err.std())
        print("--------------------------------------------------")

    def test(self):
        '''test model'''
        '''validate model and plot testing images'''
        self.model.eval()

        
        with torch.no_grad():
            for i, inputs in tqdm(enumerate(self.test_dataloader, 0)):
                if self.model.use_gt_depth:
                    inputs[0] = inputs[0].to(self.device)
                    inputs[1] = inputs[1].to(self.device)
                else:
                    inputs = inputs.to(self.device)
                losses = self.model(inputs)

                if i == 0:
                    tot_side_err = self.model.side_error_.view(-1)
                    tot_side_err_v2 = self.model.side_error_v2_.view(-1)
                    tot_mad_err = self.model.mad_error_.view(-1)
                    iter_cnt = 1
                else:
                    iter_cnt += 1
                    tot_mad_err = torch.cat([tot_mad_err, self.model.mad_error_.view(-1)], dim=0)
                    tot_side_err = torch.cat([tot_side_err, self.model.side_error_.view(-1)], dim=0)
                    tot_side_err_v2 = torch.cat([tot_side_err_v2, self.model.side_error_v2_.view(-1)], dim=0)
        
        print("--------------------------------------------------")
        print("side err mean: ", tot_side_err.mean().cpu().item(), "side err std: ", tot_side_err.std().cpu().item())
        print("side err v2 mean: ", tot_side_err_v2.mean().cpu().item(), "side err v2 std: ", tot_side_err_v2.std().cpu().item())
        print("mad err mean: ", tot_mad_err.mean().cpu().item(), "mad err std: ", tot_mad_err.std().cpu().item())
        print("--------------------------------------------------")

