import argparse
from src import Train, Test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='0',
                        help='set PCI bus id')
    parser.add_argument('--dataset', type=str, default='avenue',
                        help='datasets: ped2, avenue, shanghai')
    parser.add_argument('--epochs', type=int, default=60,
                        help='set number of epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='set batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='set number of cpu cores for dataloader')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='set learning rate')
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='set path to read/write log files')
    parser.add_argument('--data_path', type=str, default='/home/user/Documents/VADSET',
                        help='set path to datasets')
    parser.add_argument('--clip_length', type=int, default=20,
                        help='set length of a frame clip')
    parser.add_argument('--seed', type=int, default=0,
                        help='set seed for random numbers')
    parser.add_argument('--train', action='store_const', const='train')
    parser.add_argument('--inference', action='store_const', const='inference')
    args = parser.parse_args()

    if args.train:
        Train(args).run()
    elif args.inference:
        Test(args).run()