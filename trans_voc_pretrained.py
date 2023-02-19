import torch
#from torch.nn import functional as F
import sys
split=1
checkpoint = torch.load("/home/yetao/zzk/few_shot/Efficient-FSOD/Output/nosim_class/base/model_final.pth", map_location=torch.device("cpu"))
model = checkpoint['model']
aimclass = 24
change = [
          # ('roi_heads.box_predictor.bbox_pred.weight', (4,1024)),
          # ('roi_heads.box_predictor.bbox_pred.bias', 4),
          ('roi_heads.box_predictor.cls_score.weight', (aimclass, 1024)),
          ('roi_heads.box_predictor.cls_score.bias'  , aimclass),
          ]
t = torch.empty(change[0][1])
torch.nn.init.normal_(t, std=0.001)
model[change[0][0]] = t
t = torch.empty(change[1][1])
torch.nn.init.constant_(t, 0)
model[change[1][0]] = t
checkpoint = dict(model=model)
torch.save(checkpoint, 'model_final.pth')
