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

## Training & Evaluation 

To train a model, run
```angular2html
python tools/train_net.py --num-gpus 4 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_base.yaml
```

To evaluate the trained models, run
```angular2html
python tools/test_net.py --num-gpus 4 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ours_10shot.yaml \
        --eval-only
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

