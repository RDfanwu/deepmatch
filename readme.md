# Deep Match

## Instructions
This code has been tested on
- Python 3.6.13, PyTorch==1.9.0(cuda 10.2), NVIDIA Quadro P6000.
  
### Requirements
To create a virtual environment and install the required dependences please run:
```
git clone https://github.com/awuBugless/deepmatch
conda create -n deepmatch python=3.8.8
conda activate deepmatch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Datasets
The dataset ModelNet40 will be downloaded automatically.

## Train
```shell
python train.py --exp_name=clean
python train.py --exp_name=gn --gaussian_noise=True
python train.py --exp_name=unseen --unseen=True
```
## Test
```shell
python test.py --exp_name=clean --model_path="pretrained/clean.best.t7"
python test.py --exp_name=gn --gaussian_noise=True --model_path="pretrained/gn.best.t7"
python test.py --exp_name=unseen --unseen=True --model_path="pretrained/unseen.best.t7"
```