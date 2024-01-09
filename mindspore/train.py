import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.dataset as ds

import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore import nn, train
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



network = resnet50(pretrained=True)

# 全连接层输入层的大小
in_channels = network.fc.in_channels
# 输出通道数大小花卉分类数5
head = nn.Dense(in_channels, 5)
# 重置全连接层
network.fc = head

# 平均池化层kernel size为7
avg_pool = nn.AvgPool2d(kernel_size=7)
# 重置平均池化层
network.avg_pool = avg_pool

# 冻结除最后一层外的所有参数
for param in network.get_parameters():
    if param.name not in ["fc.weight", "fc.bias"]:
        param.requires_grad = False


# 定义优化器和损失函数
opt = nn.Momentum(params=network.trainable_params(), learning_rate=0.001, momentum=0.5)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

def forward_fn(inputs, targets):

    logits = network(inputs)
    loss = loss_fn(logits, targets)

    return loss

grad_fn = ms.value_and_grad(forward_fn, None, opt.parameters)

def train_step(inputs, targets):

    loss, grads = grad_fn(inputs, targets)
    opt(grads)

    return loss

# 实例化模型
model1= train.Model(network, loss_fn, opt, metrics={"Accuracy": train.Accuracy()})

# 获取处理后的训练与测试数据集

dataset_train =create_dataset(train_path)
step_size_train = dataset_train.get_dataset_size()

dataset_val = create_dataset(val_path)
step_size_val = dataset_val.get_dataset_size()

num_epochs=50

data_loader_train = dataset_train.create_tuple_iterator(num_epochs=num_epochs)
data_loader_val = dataset_val.create_tuple_iterator(num_epochs=num_epochs)

best_ckpt_path = "./resnet50-best.ckpt"


print("Start Training Loop ...")

best_acc = 0

loss_plot=[]
acc_plot=[]
for epoch in range(num_epochs):
    losses = []
    network.set_train()

    epoch_start = time.time()

    # 为每轮训练读入数据
    for i, (images, labels) in enumerate(data_loader_train):
        labels = labels.astype(ms.int32)
        loss = train_step(images, labels)
        losses.append(loss)

    # 每个epoch结束后，验证准确率

    acc = model1.eval(dataset_val)['Accuracy']

    epoch_end = time.time()
    epoch_seconds = (epoch_end - epoch_start) * 1000
    step_seconds = epoch_seconds/step_size_train
    loss_plot.append(sum(losses)/len(losses))
    acc_plot.append(acc)
    print("-" * 20)
    print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]" % (
        epoch+1, num_epochs, sum(losses)/len(losses), acc
    ))
    print("epoch time: %5.3f ms, per step time: %5.3f ms" % (
        epoch_seconds, step_seconds
    ))

    if acc > best_acc:
        best_acc = acc
        ms.save_checkpoint(network, best_ckpt_path)


x=np.arange(0,num_epochs,1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x,loss_plot, linewidth=1, linestyle="solid", label="train loss")
plt.legend()
plt.title('Loss curve')
plt.show()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x,acc_plot, 'r-',linewidth=1, linestyle="solid", label="val accuracy")
plt.legend()
plt.title('accuracy curve')
plt.show()
print("=" * 80)
print(f"End of validation the best Accuracy is: {best_acc: 5.3f}, "
      f"save the best ckpt file in {best_ckpt_path}", flush=True)


