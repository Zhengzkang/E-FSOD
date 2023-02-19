import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pylab as pl
# 得到txt文件中的数据
def getLossAcc(logFile):
    f = open(logFile, "r", encoding='utf-8')
    line = f.readline()  # 以行的形式进行读取文件
    iterate = []
    loss = []
    while line:
        nameArr = line.split(" ")
        iterate.append(int(nameArr[0]))
        loss.append(float(nameArr[1]))
        # print(accurayStr)
        line = f.readline()
    f.close()
    return iterate, loss

# 绘制曲线
def drawLine(iterate, loss, xName, yName, title):
    # 横坐标 采用列表表达式
    x = iterate
    # 纵坐标
    y = loss
    # 生成折线图：函数polt
    plt.plot(x, y)
    # 设置横坐标说明
    plt.xlabel(xName)
    # 设置纵坐标说明
    plt.ylabel(yName)
    # 添加标题
    plt.title(title)
    # 设置纵坐标刻度
    # plt.yticks(graduate)
    # 显示网格
    plt.grid(True)
    # 显示图表

    # 保存结果图
    plt.savefig("train_results_loss.png")
    plt.show()

if __name__ == '__main__':
    line = []
    with open(r"/home/yetao/zzk/few_shot/Efficient-FSOD/Output/efc/30shot/log.txt", encoding='utf-8') as f:  # 从log文件中读出数据
        for line1 in f:
            line.append(line1)

    file_handle=open('1.txt',mode='w')
    for item in line:
        # 判断每一行是否以Epoch为开头
        if(item).startswith('[11/'):
            str=[]
            strl = item.split(':')
            if len(strl)<13:
                continue
            a=strl[7]
            b=strl[8]
            a=a.split(' ')[1].strip()
            b=b.split(' ')[1].strip()
            file_handle.write(a+" "+b+'\n')
    file_handle.close()

iterate, loss = getLossAcc(r"1.txt")
drawLine(iterate, loss, "Iterated", "Loss", "Loss curve")

