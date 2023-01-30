import torch
from torch import nn
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision import transforms

from .model import Model
from .load import testloader
from .utils import label_encapsule, psnr, score_norm

import numpy as np
import sklearn.metrics as skmetr
from glob import glob
from tqdm import tqdm
import pickle
import os


class Vis():
    def __init__(self, args) -> None:
        super(Vis, self).__init__()
        self.device = torch.device(f'cuda:{args.cuda}' \
                                   if torch.cuda.is_available() \
                                   else 'cpu')
        
        self.log_path = f'{args.log_path}/{args.dataset}'
        os.makedirs(self.log_path, exist_ok=True)
        
        self.gt_label = f'{args.data_path}/{args.dataset}_gt.npy'
        self.args = args
        
    def run(self):
        pth = 'mem200_batch8_seeds0_clip20_run12-17_06:42PM_auc972.pth'
        
        print(f'test on {self.args.dataset}: {pth}...')
        
        net = Model(self.args.clip_length, self.args.dataset).to(self.device)
        net.load_state_dict(torch.load(self.log_path+'/'+pth,
                                       map_location=f'cuda:{self.args.cuda}'))
        
        MSE = nn.MSELoss().to(self.device)
        
        net.eval()
        
        flow_path = f'{self.args.data_path}/{self.args.dataset}/testing/flows'
        flows = sorted(glob(os.path.join(flow_path, '*')))
        
        frame_path = f'{self.args.data_path}/{self.args.dataset}/testing/frames'
        videos = sorted(glob(os.path.join(frame_path, '*')))
        
        labels = label_encapsule(np.load(self.gt_label).squeeze(),
                                 frame_path, self.args.clip_length)
        
        err_vid = {}
        with torch.no_grad():
            for i, (vid, fls) in enumerate(tqdm(zip(videos, flows))):
                loader = testloader(frame_path=vid,
                                    flow_path=fls,
                                    num_workers=self.args.num_workers,
                                    window=self.args.clip_length)

                err_list = []
                for idx, (frame, flow) in enumerate(loader):
                    frame = Variable(frame).to(self.device)
                    flow = Variable(flow).to(self.device)
                    
                    _, coefficients = net(frame[:,:-3], flow)
                    
                    coef = coefficients[2].detach()
                    
                    # coefficients[0] 256
                    # coefficients[1] 128
                    # coefficients[2] 64
                    # coefficients[3] 32
                    
                    vidnum = f'{i+1}'.zfill(2)
                    err_vid[vidnum] = err_list
                    
                    imglog = f'/home/user/Downloads/coefs64/{vidnum}'
                    os.makedirs(imglog, exist_ok=True)
                        
                    # np.save(f'{imglog}/{idx+1}', coef)
                    vutils.save_image(coef, f'{imglog}/{idx+1}.png')
            
            # log = '/home/user/Downloads'
            # pkl = open(f'{log}/err_{self.args.dataset}.pkl', 'wb')
            # pickle.dump(err_vid, pkl)
            # pkl.close()
            
            print('done')
