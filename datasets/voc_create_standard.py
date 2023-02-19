import os


CLASS_NAMES_split1 = [
    "BulletTrain", "Pedestrian", "RailwayStraight",
    "RailwayLeft","RailwayRight", "Helmet","Spanner",
]
# CLASS_NAMES_split1 = [
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor",
# ]
cls = CLASS_NAMES_split1[0:]
yolodir = 'voclist'
for shot in [30, 10, 5, 3, 2, 1]:
    ids = []
    for c in cls:
        with open(yolodir + '/box_%dshot_%s_train.txt'%(shot, c)) as f:
            content = f.readlines()
        content = [i.strip().split('/')[-1][:-4] for i in content]
        ids += content
    ids = list(set(ids))
    with open('VOC2007/ImageSets/Main/trainval_%dshot_novel.txt'%shot, 'w+') as f:
        for i in ids:
            if '_' not in i:
                f.write(i + '\n')
    with open('VOC2012/ImageSets/Main/trainval_%dshot_novel.txt'%shot, 'w+') as f:
        for i in ids:
            if '_' in i:
                f.write(i + '\n')

