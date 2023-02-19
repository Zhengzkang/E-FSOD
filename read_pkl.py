#!/usr/bin/env python3

# @Time  :2022/11/16 下午10:36

# @author: Zhikang Zheng
import json
import os
import pickle
path="/home/yetao/zzk/few_shot/Efficient-FSOD/Output/efc/retina_fpn/3shot"
result="/home/yetao/zzk/few_shot/Efficient-FSOD/Output/efc/retina_fpn/3shot/AP.txt"
pkl=os.path.join(path,"/home/yetao/zzk/few_shot/FSRW/output/RailwayLeft_pr.pkl")
file=open(pkl,"rb")
data=pickle.load(file)
print(data)
file.close()
for item in data.items():
    a=item[1]
    b=item[0]
    with open (result,"a+") as f:
     f.write('iter')
     f.write(str(b))
     f.write(':')
     f.write(json.dumps(a))
     f.write('\r')