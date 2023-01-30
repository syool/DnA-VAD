import torch
from torch import nn
from torch.nn import utils
from torch.autograd import Variable

from .model import Model
from .load import testloader
from .utils import label_encapsule, psnr, score_norm

import numpy as np
import sklearn.metrics as skmetr
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os


class Test():
    def __init__(self, args) -> None:
        super(Test, self).__init__()
        self.device = torch.device(f'cuda:{args.cuda}' \
                                   if torch.cuda.is_available() \
                                   else 'cpu')
        
        self.log_path = f'{args.log_path}/{args.dataset}'
        os.makedirs(self.log_path, exist_ok=True)
        
        self.gt_label = f'{args.data_path}/{args.dataset}_gt.npy'
        self.args = args
        
    def run(self):
        pth = self.args.saved
        
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
        
        # memp = net.me1.parameters()
        # for p in memp:
        #     tmp = p.cpu().detach().numpy()
            
        # tsne = TSNE(n_components=2)
        # tsnev = tsne.fit_transform(tmp)
        
        # plt.figure(figsize=(10,10))
        # plt.scatter(tsnev[:,0], tsnev[:,1])
        
        # plt.xlim(tsnev[:,0].min()-1, tsnev[:,0].max()+1)
        # plt.ylim(tsnev[:,1].min()-1, tsnev[:,1].max()+1)
        
        # plt.savefig('./run1.png')
        # print()
        
        scores = []
        with torch.no_grad():
            for i, (vid, fls) in enumerate(zip(tqdm(videos), flows)):
                loader = testloader(frame_path=vid,
                                    flow_path=fls,
                                    num_workers=self.args.num_workers,
                                    window=self.args.clip_length)

                err_list = []
                for idx, (frame, flow) in enumerate(loader):
                    frame = Variable(frame).to(self.device)
                    flow = Variable(flow).to(self.device)
                    
                    output = net(frame[:,:-3], flow)
                      
                    error = MSE(output, frame[:,-3:])
                    err_list = np.append(err_list, psnr(error.item()))
                    
                p = score_norm(err_list)
                scores = np.append(scores, p)
                # print(f'video {i+1}/{len(videos)} done')
            
        # final evaluation
        fpr, tpr, _ = skmetr.roc_curve(labels, scores, pos_label=0)
        auc = skmetr.auc(fpr, tpr)

        print(f'fianl auc: {auc}')
        
    def __call__(self):
        self.run()