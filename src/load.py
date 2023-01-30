import torch
from torch.utils import data
from torchvision import transforms

from glob import glob
from PIL import Image
import os


class Cliploader(data.Dataset):
    ''' Load a dataset using the silding window '''
    def __init__(self, frame_path, flow_path,
                 trans_frame, trans_flow, window) -> None:
        super(Cliploader, self).__init__()
        
        self.window = window
        self.trans_frame = trans_frame
        self.trans_flow = trans_flow
        self.clips_frame = self.sliding_window(frame_path, key='fr')
        self.clips_flow = self.sliding_window(flow_path, key='fl')
    
    # get windows from the entire frame set
    def sliding_window(self, data_path, key):
        videos = sorted(glob(os.path.join(data_path, '*')))
        
        entry1 = []
        for vid in videos:
            samples = sorted(glob(os.path.join(vid, '*')))
            entry1.append(samples)
        
        entry2 = []
        for vid in entry1:
            if key == 'fr':
                for i in range(len(vid)-(self.window)):
                    clip = vid[i:i+self.window+1] # +1 frame for future prediction
                    entry2.append(clip)
            elif key == 'fl':
                for i in range(len(vid)-(self.window-1)):
                    entry2.append(vid[i+self.window-1]) # one last optical flow
                
        return entry2
    
    # concat a clip
    def clipper(self, input, key):
        stack = []
        if key == 'fr':
            for i in input:
                try:
                    x = self.trans_frame(Image.open(i)) # [C, H, W]
                except Exception as e:
                    print(i)
                x = torch.squeeze(x) # [H, W]
                stack.append(x)
            cat = torch.stack(stack, axis=0) # [window, H, W]
            if len(cat.shape) == 4:
                out = cat.view(-1, 256, 256)
                
        elif key == 'fl':
            try:
                out = self.trans_flow(Image.open(input))
            except Exception as e:
                print(input)
            
        return out

    def __getitem__(self, index):
        frames = self.clips_frame[index]
        flows = self.clips_flow[index]
        
        frames = self.clipper(frames, key='fr')
        flows = self.clipper(flows, key='fl')
        
        return frames, flows

    def __len__(self):
        return len(self.clips_frame)


class Testloader(Cliploader):
    def __init__(self, frame_path, flow_path,
                 trans_frame, trans_flow, window) -> None:
        super().__init__(frame_path, flow_path,
                         trans_frame, trans_flow, window)
    
    # override -> get windows from one video
    def sliding_window(self, data_path, key):
        frames = sorted(glob(os.path.join(data_path, '*')))
        
        entry = []
        if key == 'fr':
            for i in range(len(frames)-(self.window)):
                clip = frames[i:i+self.window+1]
                entry.append(clip)
        elif key == 'fl':
            for i in range(len(frames)-(self.window-1)):
                clip = frames[i+self.window-1]
                entry.append(clip)
            
        return entry


def trainloader(frame_path, flow_path, batch, num_workers, window):
    trans_frame = [
        # transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    trans_flow = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5,0.5),(0.5,0.5,0.5,0.5))
    ]
    dataset = Cliploader(frame_path,
                         flow_path,
                         transforms.Compose(trans_frame),
                         transforms.Compose(trans_flow),
                         window)
    
    return data.DataLoader(dataset,
                           batch_size=batch,
                           shuffle=True,
                           num_workers=num_workers,
                           drop_last=True,
                           pin_memory=False)


def testloader(frame_path, flow_path, num_workers, window):
    trans_frame = [
        # transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    trans_flow = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5,0.5),(0.5,0.5,0.5,0.5))
    ]
    dataset = Testloader(frame_path,
                         flow_path,
                         transforms.Compose(trans_frame),
                         transforms.Compose(trans_flow),
                         window)
    
    return data.DataLoader(dataset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=num_workers,
                           drop_last=False,
                           pin_memory=False)