## Environments
* Python 3.8
* PyTorch 1.10.0

Import the Ananconda virtual envrionment using "env.yaml" or using Dockerfile

## Datasets
* UCSD Pedestrian 2
* CUHK Avenue
* ShanghaiTech Campus

For optical flow, we highly recommand to use the official implementation of Flownet 2.0.

## Training & Inference
```bash
# For training
python3 main.py --dataset ped2 --cuda 0 --train
```
```bash
# For inference
python3 main.py --dataset ped2 --cuda 0 --inference
```