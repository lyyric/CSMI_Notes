# 多分类

```python
%reset -f

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
np.set_printoptions(linewidth=500, precision=2, suppress=True)

torch.set_default_device("cuda")
```

## MNIST 数据集

### 导入数据

MNIST 数据集包含 70,000 张手写数字图片，每张图片大小为 28x28 像素，目标是识别这些数字。这是机器学习中的“Hello World”。

```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# 数据是经典的图像数据
X_train.shape, y_train.shape, X_train.dtype, y_train.dtype
```

查看测试集中所有唯一值：

```python
torch.unique(X_test)
```

#### 数据归一化

将数据归一化到 [0,1] 区间，使用浮点数表示，这样有助于加快训练和提高模型性能。

```python
# 使用浮点数且归一化到 [0,1]
X_train = X_train.float() / 255
X_test = X_test.float() / 255
X_train.dtype, X_test.dtype
```

展示第一张训练图片：

```python
plt.imshow(X_train[0,:,:].cpu(), cmap='gray')
plt.show()
```

### 标签处理

数据中的标签是 `uint8` 类型，这可能会导致问题。我们将其转换为 `int64` 类型，这是 PyTorch 默认的整数类型。

```python
print(y_train[:50])  # 前50个标签

# 转换标签类型
y_train = y_train.to(torch.int64)
y_test = y_test.to(torch.int64)
```

查看每个数字类别的实例数量，确认每个类别的数据量大致相同：

```python
val, count = np.unique(y_train.cpu(), return_counts=True)
plt.bar(val, count)
plt.xticks(val)
plt.xlabel('数字')
plt.ylabel('数量')
plt.show()
```

#### 可视化数据

定义一个函数，用于绘制前 25 张图片及其标签：

```python
def plot_Xy(X, y):
    ni = 5
    nj = 5
    fig, axs = plt.subplots(ni, nj, figsize=(2 * nj, 2 * ni))
    for i in range(ni):
        for j in range(nj):
            k = i * nj + j
            axs[i, j].imshow(X[k, :, :].cpu(), cmap='gray')
            axs[i, j].set_title(f"{y[k].item()}")
            axs[i, j].axis("off")
    fig.tight_layout()
    plt.show()

plot_Xy(X_train, y_train)
```

## 训练准备

### 批量数据生成器

定义一个生成器函数，用于按批次分发数据：

```python
def oneEpoch(X_all, Y_all, batch_size):
    nb_batches = len(X_all) // batch_size
    shuffle_index = np.random.permutation(len(X_all))
    X_all_shuffle = X_all[shuffle_index]
    Y_all_shuffle = Y_all[shuffle_index]

    for i in range(nb_batches):
        yield X_all_shuffle[i * batch_size:(i + 1) * batch_size], Y_all_shuffle[i * batch_size:(i + 1) * batch_size]
```

测试批量生成器：

```python
for x, y in oneEpoch(X_train, y_train, 10_000):
    print(x.shape, y.shape)
    break  # 仅打印第一个批次
```

### 代理类（Agent）

定义一个 `Agent` 类，用于管理模型的训练和验证过程：

```python
import copy

class Agent:
    def __init__(self, model, learning_rate, X_train, Y_train, batch_size, loss_fn):
        self.loss_fn = loss_fn
        self.model = model
        self.batch_size = batch_size
        nb_data_train = int(len(X_train) * 0.8)
        self.X_train = X_train[:nb_data_train]
        self.Y_train = Y_train[:nb_data_train]
        self.X_val = X_train[nb_data_train:]
        self.Y_val = Y_train[nb_data_train:]

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.losses = []
        self.val_steps = []
        self.val_losses = []
        self.step_count = -1

    def train_step(self, x, y):
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    def val_step(self, x, y):
        with torch.no_grad():
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
        return loss

    def train(self, nb_epochs):
        for epoch in range(nb_epochs):
            for x, y in oneEpoch(self.X_train, self.Y_train, self.batch_size):
                self.step_count += 1
                loss = self.train_step(x, y)
                self.losses.append(loss.detach().cpu().numpy())

            val_loss = self.val_step(self.X_val, self.Y_val).cpu().numpy()
            self.val_losses.append(val_loss)

            if val_loss <= np.min(self.val_losses):
                print(f"⬊{val_loss:.3g}", end="")
                self.best_weights = copy.deepcopy(self.model.state_dict())
            else:
                print("⬈", end="")

            self.val_steps.append(self.step_count)
        print()  # 换行

    def set_model_at_best(self):
        self.model.load_state_dict(self.best_weights)
```

## 多类分类

目标是根据图像预测其标签。

### 图像展平

将每张 2D 图像展平成 1D 向量：

```python
X_train_flat = X_train.view(-1, 28*28)
X_test_flat = X_test.view(-1, 28*28)
X_train_flat.shape
```

### 按标签绘制展平后的图像

定义一个函数，根据标签绘制前 200 张展平后的图像：

```python
def plot_one_label(label):
    X_train_one = X_train_flat[y_train == label]
    X_train_one = X_train_one[:200]
    plt.matshow(X_train_one.cpu())
    print(X_train_one.shape)
    plt.show()

plot_one_label(0)
plot_one_label(1)
plot_one_label(2)
```

从图中可以看出，即使展平后，每个标签的图像在数值上仍有其特征。

### 模型定义

我们将创建一个模型，让它自己进行图像展平并进行分类。

#### 模型结构

```python
class Model_classif(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.lay1 = torch.nn.Linear(784, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.final_layer = torch.nn.Linear(hidden_dim, 10)  # 10 类输出

    def forward(self, x):
        x = x.view(-1, 784)  # 展平
        x = self.relu1(self.lay1(x))
        x = self.relu2(self.lay2(x))
        x = self.final_layer(x)  # 输出 logits
        return x
```

测试模型：

```python
def test():
    model = Model_classif(64)
    Y_pred = model(X_train[:20])
    print(Y_pred.shape)  # 应该是 [20, 10]

test()
```

### 损失函数

手动实现交叉熵损失，确保理解其工作原理。

```python
# 示例 logits 和真实标签
y_pred_logits = torch.rand([4, 2])
y_true = torch.tensor([0, 1, 1, 0])

# 按标签索引选择预测概率
y_pred_logits[torch.arange(4), y_true]

def my_sparse_cross_entropy_from_logits(y_pred_logits, y_true):
    y_pred_proba = torch.softmax(y_pred_logits, dim=1)
    each = torch.arange(len(y_pred_proba))
    selected = y_pred_proba[each, y_true]
    return torch.mean(-torch.log(selected))

print(my_sparse_cross_entropy_from_logits(y_pred_logits, y_true))

# 使用 PyTorch 自带的 CrossEntropyLoss 进行对比
scel_torch = torch.nn.CrossEntropyLoss()
print(scel_torch(y_pred_logits, y_true))
```

输出结果应相同，验证手动实现的正确性。

### 训练模型

实例化模型和代理类，并开始训练：

```python
model = Model_classif(64)
agent = Agent(model, 1e-3, X_train, y_train, 256, torch.nn.CrossEntropyLoss())

agent.train(30)
```

### 计算准确率

完成计算准确率的步骤：

```python
# 设置为最佳模型
agent.set_model_at_best()

# 预测测试集
y_pred_logits = agent.model(X_test_flat)
y_pred = torch.argmax(y_pred_logits, dim=1)
print(y_pred.shape)  # 应为 [10000]

# 计算准确率
accuracy = torch.mean((y_pred == y_test).to(torch.float32))
print(f"准确率: {accuracy:.4f}")
```

## 错误分析

在完整的机器学习项目中，需要：

- 测试不同的数据预处理方法（重新缩放、主成分分析、去除异常值等）
- 尝试不同的模型
- 保留最有前景的模型
- 调整参数，并使用交叉验证和网格搜索或随机搜索选择最佳模型

关键在于找到自动化和手动调整之间的平衡。

### 混淆矩阵

混淆矩阵帮助分析模型在哪些类别上容易出错。

#### 定义混淆矩阵绘制函数

```python
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred,
                          normalize=True,
                          suppress_diag=False,
                          title="混淆矩阵",
                          cmap="jet",
                          precision=3):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    np.set_printoptions(precision=precision)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    classes = range(cm.shape[0])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if suppress_diag:
        np.fill_diagonal(cm, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='真实标签',
           xlabel='预测标签',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = f".{precision}f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
```

绘制混淆矩阵（隐藏对角线）：

```python
plot_confusion_matrix(y_test.cpu().numpy(), y_pred.cpu().numpy(), suppress_diag=True)
```

***练习:*** 哪些数字类别最容易被模型混淆？是否存在某种对称性？

**解答：**

观察混淆矩阵，可以发现某些数字之间存在较高的混淆率。例如，数字 `5` 和 `3` 可能容易混淆，因为它们在形状上有相似之处。同样，`8` 和 `3` 也可能被混淆。通常，形状相似或笔画数量接近的数字更容易被模型误分类。

### 可视化错误

定义一个函数，查看特定类别之间的混淆情况：

```python
def see_confusions(class_true, class_pred, X, y_true, y_pred, num_images=16):
    X_ab = X[(y_true == class_true) & (y_pred == class_pred), :, :]
    print(f"真实类别: {class_true}, 预测类别: {class_pred}, 数量: {X_ab.shape[0]}")

    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i >= len(X_ab):
            break
        ax.imshow(X_ab[i, :, :].cpu(), cmap='gray')
        ax.axis("off")
    plt.show()
```

查看特定错误分类的实例：

```python
see_confusions(8, 3, X_test, y_test, y_pred)
see_confusions(5, 3, X_test, y_test, y_pred)
see_confusions(3, 5, X_test, y_test, y_pred)
```

## 多标签分类

有时需要为同一数据赋予多个标签，例如：

- 一张照片中有多个人脸，标签可能是“有男性”、“有女性”。
- 一张照片中有多个动物，标签可能是“哺乳动物”、“昆虫”、“狮子”、“猫”、“苍蝇”等。

### 创建多标签数据

定义一个函数，将单标签转换为多标签：

```python
def make_multi_label(y):
    y_large = (y >= 7).float()        # 第一列：数字是否大于等于7
    y_odd = (y % 2 == 1).float()      # 第二列：数字是否为奇数
    y_prime = ((y == 2) | (y == 3) | (y == 5) | (y == 7)).float()  # 第三列：数字是否为质数

    y_multilabel = torch.stack([y_large, y_odd, y_prime], dim=1)
    return y_multilabel

y_multi_train = make_multi_label(y_train)
y_multi_test = make_multi_label(y_test)

print(y_multi_train[:10])
```

输出说明：

- 第一列表示数字是否大于等于7（7, 8, 9 为 1，其余为 0）
- 第二列表示数字是否为奇数
- 第三列表示数字是否为质数（2, 3, 5, 7 为 1，其余为 0）

### 数据批量生成

验证多标签数据的批量生成：

```python
for x, y in oneEpoch(X_train_flat, y_multi_train, 10_000):
    print(x.shape, y.shape)
    break
```

### 特定损失函数

定义适用于多标签分类的二元交叉熵损失：

```python
def mean_binary_crossentropy(y_pred, y_true):
    return torch.mean(-y_true * torch.log(y_pred + 1e-10) - (1 - y_true) * torch.log((1 - y_pred) + 1e-10))
```

测试损失函数：

```python
def test_bce():
    print(mean_binary_crossentropy(torch.tensor([[1., 1.]]), torch.tensor([[1., 1.]])))  # 应接近 0
    print(mean_binary_crossentropy(torch.tensor([[0., 0.]]), torch.tensor([[0., 0.]])))  # 应接近 0
    print(mean_binary_crossentropy(torch.tensor([[1., 0.]]), torch.tensor([[1., 1.]])))  # 有一项损失
    print(mean_binary_crossentropy(torch.tensor([[0., 0.]]), torch.tensor([[1., 1.]])))  # 两项损失
test_bce()
```

### 多标签模型定义

定义一个适用于多标签分类的模型：

```python
class Model_classif_multi_label(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lay1 = torch.nn.Linear(784, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.final_layer = torch.nn.Linear(hidden_dim, 3)  # 3 个标签

    def forward(self, x):
        x = x.view(-1, 784)  # 展平
        x = self.relu1(self.lay1(x))
        x = self.relu2(self.lay2(x))
        x = torch.sigmoid(self.final_layer(x))  # Sigmoid 激活，输出概率
        return x
```

测试模型输出：

```python
X = X_train_flat[:10, :]
model = Model_classif_multi_label(64)
Y_pred = model(X)
print(Y_pred.shape)            # 应为 [10, 3]
print(y_multi_train[:10, :].shape)  # [10, 3]
```

### 训练多标签模型

实例化并训练模型：

```python
agent = Agent(Model_classif_multi_label(64), 1e-3, X_train, y_multi_train, 256, mean_binary_crossentropy)
agent.train(20)
```

### 模型评估

设置为最佳模型，并进行预测：

```python
agent.set_model_at_best()

y_multi_pred = agent.model(X_test_flat)
print(y_multi_pred)

# 二值化预测结果
y_multi_pred = (y_multi_pred > 0.5).int()
print(y_multi_pred)
```

#### 计算准确率

多标签准确率的计算方式有两种：

- **宽容准确率（Gentle Accuracy）：** 每个标签独立比较，计算平均正确率。
- **严格准确率（Strict Accuracy）：** 所有标签必须完全匹配才算正确。

```python
# 宽容准确率
gentle_accuracy = torch.mean((y_multi_pred == y_multi_test).float())
print(f"宽容准确率: {gentle_accuracy:.4f}")

# 严格准确率
both = (y_multi_pred[:,0] == y_multi_test[:,0]) & \
       (y_multi_pred[:,1] == y_multi_test[:,1]) & \
       (y_multi_pred[:,2] == y_multi_test[:,2])
strict_accuracy = torch.mean(both.float())
print(f"严格准确率: {strict_accuracy:.4f}")
```

***练习:*** 用几句话解释“宽容准确率”和“严格准确率”是什么意思。

**解答：**

- **宽容准确率（Gentle Accuracy）：** 这是指每个标签独立地进行比较，然后取所有标签的平均正确率。即，模型对每个标签的预测是否正确不影响其他标签。
  
- **严格准确率（Strict Accuracy）：** 这是指只有当模型对所有标签的预测都正确时，才算一次正确预测。即，所有标签都必须完全匹配，才能被认为是正确的。

## 多输出分类

多输出分类涵盖所有输出 `Y` 较为复杂的情况。

### 应用示例

- **示例 1：** 输入 `X` 是一张图片，输出 `Y` 也是一张图片，表示 `X` 的分割结果。例如，`Y[i,j]` 表示 `X[i,j]` 是否属于道路、房屋、人类或树木。
  
- **示例 2：** 输入 `X` 是车辆拍摄的图像，输出 `Y` 是一个包含分割结果和车辆是否可以前进的变量的组合。

- **示例 3：** 前面的多标签分类可以看作是一种多输出分类，我们有多个输出，每个输出描述了一个属性。

### 具体案例：图像去噪

输入 `X` 是加噪声的 MNIST 图像，输出 `Y` 是去噪后的图像。

#### 创建数据

定义一个函数，为图像添加噪声：

```python
def add_noise(X):
    noise = torch.rand(len(X), 28, 28) * 2
    noise2 = 0.9 * ((torch.rand(len(X), 28, 28) < 0.2).float())
    return X + noise + noise2

X_train_mo = add_noise(X_train)
X_test_mo = add_noise(X_test)

# 标签是去噪后的二值化图像
Y_train_mo = (X_train > 0).float()
Y_test_mo = (X_test > 0).float()

def plot_imgs(imgs_list, labels):
    ni = len(imgs_list)
    nj = 10

    fig, axs = plt.subplots(ni, nj, figsize=(nj * 2, ni * 2))
    for i, imgs in enumerate(imgs_list):
        for j in range(nj):
            axs[i, j].imshow(imgs[j, :, :].cpu(), cmap='gray')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    for i, label in enumerate(labels):
        axs[i, 0].set_ylabel(label)
    plt.show()

plot_imgs([X_train_mo, Y_train_mo], ["输入", "输出"])
```

#### 多输出模型定义

```python
class Model_classif_multi_output(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lay1 = torch.nn.Linear(784, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.final_layer = torch.nn.Linear(hidden_dim, 784)  # 输出与输入同尺寸

    def forward(self, x):
        x = x.view(-1, 784)  # 展平
        x = self.relu1(self.lay1(x))
        x = self.relu2(self.lay2(x))
        x = torch.sigmoid(self.final_layer(x))  # Sigmoid 激活，输出概率
        x = x.view(-1, 28, 28)  # 还原为图片形状
        return x
```

#### 训练多输出模型

```python
model_mo = Model_classif_multi_output(256)
agent_mo = Agent(model_mo, 1e-3, X_train_mo, Y_train_mo, 256, mean_binary_crossentropy)

agent_mo.train(30)
```

#### 模型预测与可视化

```python
agent_mo.set_model_at_best()

Y_pred_mo = agent_mo.model(X_test_mo)
Y_pred_mo_bin = (Y_pred_mo > 0.5).int()

plot_imgs([X_test_mo, Y_test_mo, Y_pred_mo, Y_pred_mo_bin], ["输入", "输出", "预测", "二值化预测"])
```

## 进阶练习

### 去噪问题的回归变体

将去噪问题视为回归问题更为自然，可以使用均方误差（MSE）作为损失函数，并调整模型的激活函数。

#### 定义 MSE 损失

```python
def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
```

***练习:*** 创建一个模型，并选择合适的损失函数，以从去噪输入 `X_test_mo` 恢复原始图像 `X_test`。

**解答：**

需要定义一个新的模型（可以类似于 `Model_classif_multi_output`），但输出层不使用 Sigmoid 激活函数，因为回归任务通常输出连续值。损失函数选择 MSE。

```python
class Model_regression(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lay1 = torch.nn.Linear(784, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.final_layer = torch.nn.Linear(hidden_dim, 784)  # 输出与输入同尺寸

    def forward(self, x):
        x = x.view(-1, 784)  # 展平
        x = self.relu1(self.lay1(x))
        x = self.relu2(self.lay2(x))
        x = self.final_layer(x)  # 线性输出
        x = x.view(-1, 28, 28)  # 还原为图片形状
        return x
```

训练模型：

```python
model_reg = Model_regression(256)
agent_reg = Agent(model_reg, 1e-3, X_train_mo, X_train, 256, mse)  # 使用 MSE 作为损失函数

agent_reg.train(30)
```

预测与可视化：

```python
agent_reg.set_model_at_best()

Y_pred_reg = agent_reg.model(X_test_mo)

# 计算回归准确率（可以使用 MSE 或其他回归指标）
test_mse = mse(Y_pred_reg, X_test)
print(f"测试集 MSE: {test_mse.item():.4f}")

# 可视化部分结果
def plot_regression_results(X_input, Y_true, Y_pred, num_images=10):
    fig, axs = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
    for i in range(num_images):
        axs[0, i].imshow(X_input[i].cpu(), cmap='gray')
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_ylabel('输入')

        axs[1, i].imshow(Y_true[i].cpu(), cmap='gray')
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_ylabel('真实输出')

        axs[2, i].imshow(Y_pred[i].detach().cpu(), cmap='gray')
        axs[2, i].axis('off')
        if i == 0:
            axs[2, i].set_ylabel('预测输出')
    plt.tight_layout()
    plt.show()

plot_regression_results(X_test_mo, X_test, Y_pred_reg, num_images=10)
```

### 复杂练习

#### 多标签硬问题

定义一个更复杂的多标签问题，要求模型从混合图像中预测多个标签。

```python
def make_multi_label_hard(X, y):
    n = len(X)

    rand_perm = np.random.permutation(n)
    X1 = X[rand_perm]
    y1 = y[rand_perm]

    rand_perm = np.random.permutation(n)
    X2 = X[rand_perm]
    y2 = y[rand_perm]

    rand_perm = np.random.permutation(n)
    X3 = X[rand_perm]
    y3 = y[rand_perm]

    rand_perm = np.random.permutation(n)
    X4 = X[rand_perm]
    y4 = y[rand_perm]

    Z_haut = torch.cat([X1, X2], dim=1)  # 横向拼接
    Z_bas = torch.cat([X3, X4], dim=1)
    Z = torch.cat([Z_haut, Z_bas], dim=2)  # 纵向拼接

    yy = torch.stack([y1, y2, y3, y4], dim=1)
    for i in range(n):
        yy[i, :] = torch.sort(yy[i, :])[0]

    return Z, yy

X_train_mlh, Y_train_mlh = make_multi_label_hard(X_train, y_train)
plot_Xy(X_train_mlh, Y_train_mlh)
```

#### 多标签硬问题训练

***练习:*** 从混合图像中尽可能准确地预测标签。训练过程应不太耗时，关键在于如何“编码”这些标签。

**解答：**

首先，定义合适的模型。由于每个输出有多个标签，可以使用多输出模型，每个标签使用独立的输出单元。

```python
class Model_classif_multi_label_hard(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lay1 = torch.nn.Linear(28*28, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.final_layer = torch.nn.Linear(hidden_dim, 4)  # 假设有4个标签

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平
        x = self.relu1(self.lay1(x))
        x = self.relu2(self.lay2(x))
        x = torch.sigmoid(self.final_layer(x))  # Sigmoid 激活
        return x
```

定义新的损失函数（多标签交叉熵）：

```python
def multi_label_cross_entropy(y_pred, y_true):
    return torch.mean(-y_true * torch.log(y_pred + 1e-10) - (1 - y_true) * torch.log((1 - y_pred) + 1e-10))
```

实例化并训练模型：

```python
model_mlh = Model_classif_multi_label_hard(256)
agent_mlh = Agent(model_mlh, 1e-3, X_train_mlh, Y_train_mlh, 256, multi_label_cross_entropy)

agent_mlh.train(30)
```

预测与评估：

```python
agent_mlh.set_model_at_best()

Y_pred_mlh = agent_mlh.model(X_test_mo)
Y_pred_mlh_bin = (Y_pred_mlh > 0.5).int()

# 计算宽容和严格准确率
gentle_accuracy_mlh = torch.mean((Y_pred_mlh_bin == Y_train_mlh).float())
both_mlh = torch.all(Y_pred_mlh_bin == Y_train_mlh, dim=1)
strict_accuracy_mlh = torch.mean(both_mlh.float())

print(f"宽容准确率: {gentle_accuracy_mlh:.4f}")
print(f"严格准确率: {strict_accuracy_mlh:.4f}")
```

# 总结

在本笔记中，我们探讨了多类分类、多标签分类和多输出分类的基本概念和实现方法。通过使用 MNIST 数据集，我们展示了如何进行数据预处理、模型定义、损失函数选择、模型训练以及错误分析。通过练习和实际操作，加深了对这些概念的理解和应用。

# 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn 文档](https://scikit-learn.org/stable/documentation.html)
- [TensorFlow Keras MNIST 数据集](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist)

# 结束语

希望这份笔记能够帮助你更好地理解多分类任务的各个方面，并在实际项目中应用这些知识。如果有任何疑问，欢迎随时讨论！