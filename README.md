# clipreid-pro
CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels 
PWC

Pipeline
framework

Installation
conda create -n clipreid python=3.8
conda activate clipreid
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs
pip install timm
pip install scikit-image
pip install tqdm
pip install ftfy
pip install regex
Prepare Dataset
Download the datasets (Market-1501, MSMT17, DukeMTMC-reID, Occluded-Duke, VehicleID, VeRi-776), and then unzip them to your_dataset_dir.

Training
For example, if you want to run CNN-based CLIP-ReID-baseline for the Market-1501, you need to modify the bottom of configs/person/cnn_base.yml to

DATASETS:
   NAMES: ('market1501')
   ROOT_DIR: ('your_dataset_dir')
OUTPUT_DIR: 'your_output_dir'
then run

CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/person/cnn_base.yml
if you want to run ViT-based CLIP-ReID for MSMT17, you need to modify the bottom of configs/person/vit_clipreid.yml to

DATASETS:
   NAMES: ('msmt17')
   ROOT_DIR: ('your_dataset_dir')
OUTPUT_DIR: 'your_output_dir'
then run

CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/person/vit_clipreid.yml
if you want to run ViT-based CLIP-ReID+SIE+OLP for MSMT17, run:

CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/person/vit_clipreid.yml  MODEL.SIE_CAMERA True MODEL.SIE_COE 1.0 MODEL.STRIDE_SIZE '[12, 12]'
Evaluation
For example, if you want to test ViT-based CLIP-ReID for MSMT17



Trained models and test logs
Datasets	MSMT17	Market	Duke	Occ-Duke	VeRi	VehicleID
