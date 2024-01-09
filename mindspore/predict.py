
import os
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn, train
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore import Model, Tensor, context, save_checkpoint, load_checkpoint, load_param_into_net
from model import resnet50

#设置使用设备，CPU/GPU/Ascend
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
train_path=os.path.join(image_path, "train")
val_path=os.path.join(image_path, "val")

def create_dataset(data_path, batch_size=16, repeat_num=1, training=True):
    """定义数据集"""
    data_set = ds.ImageFolderDataset(data_path, num_parallel_workers=8, shuffle=True)

    image_size = [224, 224]
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    trans = [
        CV.Decode(),
        CV.Resize(image_size),
        CV.Normalize(mean=mean, std=std),
        CV.HWC2CHW()
    ]

    # 实现数据的map映射、批量处理和数据重复的操作
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set

dataset_train =create_dataset(train_path)
step_size_train = dataset_train.get_dataset_size()

dataset_val = create_dataset(val_path)
step_size_val = dataset_val.get_dataset_size()

best_ckpt_path = "./resnet50-best.ckpt"
def visualize_model(best_ckpt_path, val_ds):
    net = resnet50()
    # 全连接层输入层的大小
    in_channels = net.fc.in_channels
    head = nn.Dense(in_channels, 5)
    # 重置全连接层
    net.fc = head
    # 平均池化层kernel size为7
    avg_pool = nn.AvgPool2d(kernel_size=7)
    # 重置平均池化层
    net.avg_pool = avg_pool
    # 加载模型参数
    param_dict = ms.load_checkpoint(best_ckpt_path)
    ms.load_param_into_net(net, param_dict)
    model = train.Model(net)
    # 加载验证集的数据进行验证
    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    class_name = {0: "daisy",
    1: "dandelion",
    2: "roses",
    3: "sunflowers",
    4: "tulips"}
    # 预测图像类别
    output = model.predict(ms.Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)

    # 显示图像及图像的预测值
    plt.figure(figsize=(5, 5))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        # 若预测正确，显示为蓝色；若预测错误，显示为红色
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title('predict:{}'.format(class_name[pred[i]]), color=color)
        picture_show = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        picture_show = std * picture_show + mean
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.axis('off')

    plt.show()

visualize_model(best_ckpt_path, dataset_val)