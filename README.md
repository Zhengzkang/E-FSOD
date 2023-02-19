# Installation
## Requirements
Linux with Python >= 3.6
PyTorch >= 1.4
torchvision that matches the PyTorch installation
CUDA 9.2, 10.0, 10.1, 10.2, 11.0
GCC >= 4.9
detectron
## install pytorch and cuda here https://pytorch.org/get-started/previous-versions/
```
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
## download our repo and install detectron2 (for ubuntu, must correspond to pytorch)
install detectron2 here https://link.csdn.net/?target=https%3A%2F%2Fdetectron2.readthedocs.io%2Fen%2Flatest%2Ftutorials%2Finstall.html
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
 

cd E-FSOD && pip install -e .
```
## install pycocotools, lvis-api and tensorboard
```
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install tensorboard
```
To __rebuild__ detectron2, `rm -rf build/ **/*.so` then `pip install -e .`.
You often need to rebuild detectron2 after reinstalling PyTorch.

## Data Prepare
###  for VOC
We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits.(soft link in datasets/VOC)
The shots in the datasets/vocsplit directory are the same shots used by previous works(TFA). run voc_create_base.py and voc_create_standard.py (note path of data)
### for custom 
You must 
## Training, Evaluation & Visualize

Base training
To train a model, run
```
python tools/train_net.py --num-gpus 4 \
        --config-file configs/VOC/faster_rcnn_R_101_FPN_base.yaml
```

Fine-tune
```
python tools/train_net.py --num-gpus 4 \
        --config-file configs/VOC/faster_rcnn_R_101_FPN_ours_split1_10shot.yaml
```
modify 10shot to 1,2,3,5,10 for all setting.


To evaluate the trained models, run
```
python tools/test_net.py --num-gpus 4 \
        --config-file configs/VOC/faster_rcnn_R_101_FPN_ours_10shot.yaml \
        --eval-only
```
To Vis the image/video/camer
```
python demo.py --config-file configs/VOC/faster_rcnn_R_101_FPN_ours_10shot.yaml \
               --input/--video-input/--webcam file \
	       --output outpu_dir
```

# Citing E-FSOD

```BibTeX
@article{yang2022efficient,
  title={Efficient Few-Shot Object Detection via Knowledge Inheritance},
  author={Yang, Ze and Zhang, Chi and Li, Ruibo and Lin, Guosheng},
  journal={arXiv preprint arXiv:2203.12224},
  year={2022}
}
@inproceedings{sun2021fsce,
	title={Fsce: Few-shot object detection via contrastive proposal encoding},
	author={Sun, Bo and Li, Banghuai and Cai, Shengcai and Yuan, Ye and Zhang, Chi},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	pages={7352--7362},
	year={2021}
}
```

# License

This repository is released under the [Apache 2.0 license](LICENSE).

