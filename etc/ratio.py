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

from .error import ssim


class Ratio():
    def __init__(self, args) -> None:
        super(Ratio, self).__init__()
        self.device = torch.device(f'cuda:{args.cuda}' \
                                   if torch.cuda.is_available() \
                                   else 'cpu')
        
        self.log_path = f'{args.log_path}/{args.dataset}'
        os.makedirs(self.log_path, exist_ok=True)
        
        self.gt_label = f'{args.data_path}/{args.dataset}_gt.npy'
        self.args = args
        
    def run(self):
        pth = 'mem200_clip20_ep15_auc868.pth'
        
        print(f'test on {self.args.dataset}: {pth}...')
        
        net = Model(self.args.clip_length).to(self.device)
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
                    
                    output = net(frame[:,:-3], flow)
                    
                    tr = transforms.Compose([
                         transforms.ToPILImage(),
                         transforms.Grayscale(),
                         transforms.ToTensor()
                    ])
                    
                    out = tr((output[0].detach().cpu()+1)/2)
                    target = tr((frame[0,-3:].detach().cpu()+1)/2)
                    
                    out = out.unsqueeze(1)
                    target = target.unsqueeze(1)
                    
                    error = 1 - ssim(out, target, data_range=1)
                    error = error.cpu().detach().numpy().squeeze(1)
                    
                    err_list.append(error)
                
                vidnum = f'{i+1}'.zfill(2)
                err_vid[vidnum] = err_list
                
                # imglog = f'/home/user/Downloads/errors/{vidnum}'
                # os.makedirs(imglog, exist_ok=True)
                    
                # np.save(f'{imglog}/{idx+1}', error)
                # vutils.save_image(error, f'{imglog}/{idx+1}.png')
            
            log = '/home/user/Downloads'
            pkl = open(f'{log}/err_{self.args.dataset}.pkl', 'wb')
            pickle.dump(err_vid, pkl)
            pkl.close()
            
            print('done')
