'''
under implementation
'''
import pstats
import torch
import torch.optim as optims
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os.path as path
import os

from unsup3d.model import PhotoGeoAE
from unsup3d.dataloader import CelebA, BFM


# initially, 
LR = 1e-4
max_epoch = 200
load_chk = False
chk_PATH = './chk.pt'   # need to change later

is_debug = False

class Trainer():
    def __init__(self, configs, model = None): # model is for debug(05/09)
        '''initialize params (to be implemented)'''
        self.max_epoch = configs['num_epochs']
        self.img_size = configs['img_size']
        self.batch_size = configs['batch_size']
        self.learning_rate = configs['learning_rate']

        self.epoch = 0
        self.best_loss = 1e10

        '''path relevant'''
        self.exp_name = configs['exp_name']
        self.exp_path = path.join(configs['exp_path'], self.exp_name)
        os.makedirs(self.exp_path, exist_ok=True)
        self.save_path = path.join(self.exp_path, 'models')
        os.makedirs(self.save_path, exist_ok=True)
        self.best_path = path.join(self.save_path, 'best.pt')
        self.load_path = None

        '''logger setting'''
        # self.writer = SummaryWriter('runs/fashion_mnist_experiment_1')
        self.writer = SummaryWriter(path.join(self.exp_path, 'logs'))

        '''implement dataloader'''
        if configs['dataset'] == "celeba":
            self.datasets = CelebA()
        elif configs['dataset'] == "bfm":
            self.datasets = BFM()

        self.dataloader = DataLoader(
            self.datasets,
            batch_size= self.batch_size,
            shuffle=True,
            num_workers=4
        )

        '''select GPU'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        '''define optimizer and scheduler'''
        if is_debug:
            self.model = model.to(self.device)                                  #### to debug
        else:
            self.model = PhotoGeoAE(configs).to(self.device)
        
        self.optimizer = optims.Adam(
            params = self.model.parameters(),
            lr = self.learning_rate
        )

        self.scheduler = optims.lr_scheduler.LambdaLR(
            optimizer = self.optimizer,
            lr_lambda = lambda epoch: 0.95 ** epoch
        )

        '''load_model and optimizer state'''
        if load_chk:
            self.load_model(self.load_path if self.load_path is not None else self.best_path)
        

    def train(self):
        init_epch = self.epoch
        for epch in range(init_epch, self.max_epoch):
            epch_loss = self._train()
            self.epoch = epch

            if epch_loss < self.best_loss:
                self.save_model(epch_loss)
                self.best_loss = epch_loss

            if self.epoch % 20 == 0:
                self.save_model(epch_loss)

            '''add results to tensorboard'''
            self.model.visualize(epch)
            
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
            
            losses = self.model(inputs)
            loss = torch.mean(losses)
            loss.backward()
            self.optimizer.step()

            # calculate epch_loss
            epch_loss += loss.detach().cpu()
            cnt += 1

            if i % 30 == 0:
                print(i, "step, loss : ", loss.detach().cpu().item())
        
        self.scheduler.step()
        return epch_loss

    def load_model(self, PATH):
        '''
        save loaded model (05/08)
        '''
        chkpt = torch.load(PATH)
        self.model.load_state_dict(chkpt['model_state_dict'])
        self.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        self.epoch = chkpt['epoch']
        self.best_loss = chkpt['loss']

    def save_model(self, loss):
        if loss < self.best_loss:
            PATH = path.join(self.save_path, 'best.pt')
        else:
            # saving chkpts periodically
            PATH = path.join(self.save_path, 'epoch_'+str(self.epoch)+'.pt')
        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, PATH)


    def _test(self):
        '''test model'''
        pass