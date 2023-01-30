import os

dataset = 'avenue'
log_path = './logs'

index = sorted(os.listdir(log_path+'/'+dataset))
print('DATASET:', dataset)

for i in range(len(index)):
    if i > 21:
        launcher = f'python3 main.py --cuda 1 --dataset {dataset} --test --saved {index[i]}'
        os.system(launcher)