import argparse
from src import Train, Test, Vis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # cuda settings
    parser.add_argument('--cuda', type=str, default='0', help='PCI bus id')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='cudnn.benchmark')
    parser.add_argument('--cudnn_deterministic', type=bool, default=False, help='cudnn.deterministic')
    
    # optimization settings
    parser.add_argument('--dataset', type=str, default='ped2', help='datasets: ped2, avenue, shanghai')
    parser.add_argument('--epochs', type=int, default=100, help='set number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='set batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='set learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='set number of cpu cores for dataloader')
    parser.add_argument('--clip_length', type=int, default=20, help='set length of a frame clip')
    parser.add_argument('--seed', type=int, default=0, help='set seed for random numbers')
    
    # directory settings
    parser.add_argument('--data_path', type=str, default='/root/VADSET', help='set path to datasets')
    parser.add_argument('--log_path', type=str, default='./logs', help='set path to read/write log files')
    parser.add_argument('--saved', type=str, default='auc972_5-4-3.pth', help='saved model')
    
    # switches
    parser.add_argument('--train', action='store_const', const='train')
    parser.add_argument('--test', action='store_const', const='test')
    parser.add_argument('--vis', action='store_const', const='vis')
    
    args = parser.parse_args()
    train, test = Train(args), Test(args)

    if args.train: train()
    elif args.test: test()