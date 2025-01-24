import pandas as pd
import numpy as np
import torch

#将class_dict.csv中的数据存放在字典中并返回该字典
def get_label_info(csv_path="./datasets/data/class_dict.csv"):
    data = pd.read_csv(csv_path)
    label = {}
    for _, row in data.iterrows():
        name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[name] = [int(r), int(g), int(b)]
    return label

#将三通道的label转换成二通道的分类
def one_hot_it(label, label_info = get_label_info):
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
    image = image.permute(1, 2, 0)  # [2, 512, 512] ==> [512, 512, 2]
    x = torch.argmax(image, dim=-1)  # 返回目标维度上最大值的索引
    return x

if __name__=='__main__':
    x = torch.rand([3, 5, 5])
    out = one_hot_it(x)
    print(x)
    print(get_label_info())
    print("****************************************")
