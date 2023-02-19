#!/usr/bin/env python3

# @Time  :2022/11/3 下午7:01

# @author: Zhikang Zheng
# import torch
#
# pthfile = r'/home/yetao/zzk/few_shot/Efficient-FSOD/Output/FPN/base/model_final.pth'  #faster_rcnn_ckpt.pth
# # pthfile = r'model_final.pth'  #faster_rcnn_ckpt.pth
# net = torch.load(pthfile,map_location=torch.device('cpu')) # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上
#
# print(type(net))  # 类型是 dict
# print(len(net))   # 长度为 4，即存在四个 key-value 键值对
#
# for k in net.keys():
#     print(k)      # 查看四个键，分别是 model,optimizer,scheduler,iteration
#
# for key,value in net["model"].items():
#     print(key,value.size(),sep="   ")
import torch

path = '/home/yetao/zzk/few_shot/Efficient-FSOD/Output/efc/retina_fpn/5shot/model_0001299.pth'
model = torch.load(path)

print(type(model))
print(len(model))

for k in model.keys():
    print(k)

for key, value in model['model'].items():
    print(key,value.size(),sep="   ")