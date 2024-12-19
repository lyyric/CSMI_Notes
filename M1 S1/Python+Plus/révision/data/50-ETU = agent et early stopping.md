## 数据

```python
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
```

### 加载数据

```python
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

我们不保留所有的图片，只保留部分数据。

```python
keept = 8_000
train_images = train_images[:keept]
train_labels = train_labels[:keept]

train_images.shape
```

#### ♡♡

**练习:** 这些是图片，所以请将它们显示出来！

```python
# 显示前16张训练图片
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()
```

### 预处理

```python
X_train = np.reshape(train_images / 255.0, [-1, 28*28])
X_test = np.reshape(test_images / 255.0, [-1, 28*28])

Y_train = train_labels
Y_test = test_labels

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)

Y_train_torch = torch.tensor(Y_train, dtype=torch.int64)
Y_test_torch = torch.tensor(Y_test, dtype=torch.int64)

# 如果要在GPU上运行，可以添加 `.to("cuda")` 例如：
# X_train_torch = torch.tensor(X_train, dtype=torch.float32).to("cuda")
```

### 批次分配器

* “step” 通常指优化器在一个“批次”数据上执行的一步优化。
* “epoch” 指的是多个“step”的大集合。在每个epoch结束时，会进行一次验证步骤，即在未被优化器见过的验证数据上观察损失（loss）和其他指标。

* 有时一个epoch包含任意数量的step。
* 但当数据量有限时，传统上一个epoch对应于所有数据通过一次。在这种情况下，每个epoch的step数量大约是 `数据量 // 批次大小`。

以下程序将所有数据划分为批次。

```python
def oneEpoch(X_all, Y_all, batch_size):
    nb_batches = len(X_all) // batch_size

    shuffle_index = np.random.permutation(len(X_all))
    X_all_shuffle = X_all[shuffle_index]
    Y_all_shuffle = Y_all[shuffle_index]

    for i in range(nb_batches):
        yield X_all_shuffle[i*batch_size:(i+1)*batch_size], Y_all_shuffle[i*batch_size:(i+1)*batch_size]
```

**Python小贴士:** 当有重复的切片操作时，为了避免代码重复，可以这样定义：

```python
for i in range(nb_batches):
    sl = slice(i*batch_size, (i+1)*batch_size)
    yield X_all_shuffle[sl], Y_all_shuffle[sl]
```

注意，每个epoch开始时数据都会被随机打乱，这样优化器每次处理数据的顺序不同。

注意关键字 `yield`，它用于创建一个迭代器。这类似于在循环中逐步返回结果。然后可以这样使用迭代器：

```python
# 创建迭代器
batch_dealer = oneEpoch(X_train, Y_train, 256)
# 使用迭代器
for X_batch, Y_batch in batch_dealer:
    print(X_batch.shape, Y_batch.shape)
```

### 不丢失数据的批次分配器

**注意:** 使用我们的技术，所有批次的大小都完全相同，但如果数据量不能被 `batch_size` 整除，最后会丢失一些数据。

为了避免这种情况，可以这样做：

```python
def oneEpoch_sansPerte(X_all, Y_all, batch_size):
    nb_batches = len(X_all) // batch_size + 1

    shuffle_index = np.random.permutation(len(X_all))
    X_all_shuffle = X_all[shuffle_index]
    Y_all_shuffle = Y_all[shuffle_index]

    for i in range(nb_batches):
        yield X_all_shuffle[i*batch_size:(i+1)*batch_size], Y_all_shuffle[i*batch_size:(i+1)*batch_size]
```

在这种情况下，最后一个批次的大小会比其他批次小。但是有时这会导致一个烦人的 `nan` 错误（参见下面的测试）。

**练习:** 分析并修改 `oneEpoch_sansPerte` 以避免这个错误。

```python
def oneEpoch_sansPerte_corrected(X_all, Y_all, batch_size):
    nb_batches = len(X_all) // batch_size
    remainder = len(X_all) % batch_size

    shuffle_index = np.random.permutation(len(X_all))
    X_all_shuffle = X_all[shuffle_index]
    Y_all_shuffle = Y_all[shuffle_index]

    for i in range(nb_batches):
        yield X_all_shuffle[i*batch_size:(i+1)*batch_size], Y_all_shuffle[i*batch_size:(i+1)*batch_size]
    
    if remainder != 0:
        yield X_all_shuffle[nb_batches*batch_size:], Y_all_shuffle[nb_batches*batch_size:]
```

这样，最后一个批次只有在有剩余数据时才生成，并且批次大小可能较小，从而避免 `nan` 错误。

```python
def test(batch_size):
    data_size = 100
    X = torch.rand(data_size, 3)
    Y = torch.rand(data_size, 2)

    model = torch.nn.Linear(3, 2)

    ite = oneEpoch_sansPerte_corrected(X, Y, batch_size)

    for x, y in ite:
        y_pred = model(x)
        loss = torch.mean((y - y_pred)**2)
        print(loss)

test(23)
test(20)
```

## 训练

### 一个训练代理

代理是一个用于训练模型的对象。

```python
import copy

class Agent:

    def __init__(self, model, learning_rate, X, Y, batch_size):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model = model
        self.batch_size = batch_size
        nb_data_train = int(len(X) * 0.8)
        self.X_train = X[:nb_data_train]
        self.Y_train = Y[:nb_data_train]
        self.X_val = X[nb_data_train:]
        self.Y_val = Y[nb_data_train:]

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
        for _ in range(nb_epochs):
            for x, y in oneEpoch(self.X_train, self.Y_train, self.batch_size):
                self.step_count += 1
                loss = self.train_step(x, y)
                self.losses.append(loss.detach().cpu().numpy())

            val_loss = self.val_step(self.X_val, self.Y_val)
            self.val_losses.append(val_loss.detach().cpu().numpy())

            if val_loss <= np.min(self.val_losses):
                print(f"⬊{val_loss:.2f}", end="")
                self.best_weights = copy.deepcopy(self.model.state_dict())
            else:
                print("⬈", end="")

            self.val_steps.append(self.step_count)

    def set_model_at_best(self):
        self.model.load_state_dict(self.best_weights)
```

### 模型

#### →♡

**练习:** 完善以下模型类，使其适用于分类任务。

```python
class Model_classif(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lay1 = torch.nn.Linear(784, hidden_dim)
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lay3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.final_layer = torch.nn.Linear(hidden_dim, 10)  # 10 类别

    def forward(self, x):
        x = torch.relu(self.lay1(x))
        x = torch.relu(self.lay2(x))
        x = torch.relu(self.lay3(x))
        return self.final_layer(x)

def test():
    X = torch.rand(1, 784)
    model = Model_classif(50)
    Y_pred = model(X)
    print(Y_pred.shape)

test()
```

输出应为：

```
torch.Size([1, 10])
```

### 开始训练

```python
batch_size = 256
learning_rate = 1e-3
hidden_dim = 50
model = Model_classif(hidden_dim)

agent = Agent(model, learning_rate, X_train_torch, Y_train_torch, batch_size)

agent.train(100)

plt.plot(agent.losses, label='Training Loss')
plt.plot(agent.val_steps, agent.val_losses, "r.", label='Validation Loss')
plt.legend()
plt.show()
```

```python
def accuracy_test(model):
    with torch.no_grad():
        Y_pred_logits = model(X_test_torch).cpu().numpy()
        Y_pred = np.argmax(Y_pred_logits, axis=1)
    return np.mean(Y_pred == Y_test)

def loss_test(model):
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        Y_pred_proba = model(X_test_torch)
    return loss_fn(Y_pred_proba, Y_test_torch).cpu().numpy()

print("测试集上的损失:", loss_test(agent.model))
print("测试集上的准确率:", accuracy_test(agent.model))

agent.set_model_at_best()
print("最佳模型在测试集上的损失:", loss_test(agent.model))
print("最佳模型在测试集上的准确率:", accuracy_test(agent.model))
```

**注意:** 有时“最佳”模型的准确率并不比训练结束时的模型高。准确率和损失之间只有间接的关联。

**练习:** 不重新定义模型，运行 `agent.train(50)` 第二次。你将获得如下的损失曲线。解释原因。

```python
# 重新训练代理
agent.train(50)

# 绘制新的损失曲线
plt.plot(agent.losses, label='Training Loss')
plt.plot(agent.val_steps, agent.val_losses, "r.", label='Validation Loss')
plt.legend()
plt.show()
```

**解释:** 第二次训练时，模型已经加载了之前的最佳权重。如果不重新初始化模型，继续训练可能导致模型过拟合，验证损失可能不会继续降低，甚至可能上升。这是因为模型开始记住训练数据的细节，而不是泛化到未见过的数据。

## 对抗过拟合

在训练模型的过程中，我们会面临优化和泛化之间的紧张关系。

* “优化”使模型更好地拟合当前训练数据。
* “泛化”使模型能够适应未来将要处理的数据，即模型在未见过的数据上表现良好。为了模拟这些数据，我们在训练过程中使用验证集，并在训练结束时使用测试集。

在训练初期，优化和泛化会同时提高：训练集和验证集的损失都会下降。但在训练多次之后，泛化性能不再提高，验证集的指标开始恶化：模型开始过拟合，即模型开始学习训练数据中的特定细节，而这些细节对于未来的数据并不重要。

“模型正则化”就是“对抗过拟合”的同义词。正则化意味着平滑模型，忽略细节。

我们来看看一些正则化技术。

### 提前停止（Early Stopping）

这是必须系统性实施的技术：

每次进行验证时，我们计算 `val_loss`。如果 `val_loss` 达到一个新低，我们就保存模型的参数。在训练结束时，我们将模型的参数设置为那些 `val_loss` 最低时的参数。

**变体:** 我们可以保存最近的 n 个最佳参数（例如 n=3）。然后在训练结束时，将这些“好”参数的平均值赋给模型。例如，假设有一个两层的神经网络：

$$
\text{Model}(x) =   (\sigma(x \cdot w + b)) \cdot w' + b'
$$

我们保存了 3 个记录的参数：

$$
(w_t, w'_t, b_t, b'_t)_{t \in 0,1,2}
$$

最终模型为：

$$
\text{Model}(x) =   (\sigma(x \cdot \bar{w} + \bar{b})) \cdot \bar{w}' + \bar{b}'
$$

其中，

$$
\bar{w} = \frac{w_0 + w_1 + w_2}{3}, \quad \bar{w}' = \frac{w'_0 + w'_1 + w'_2}{3}, \quad \bar{b} = \frac{b_0 + b_1 + b_2}{3}, \quad \bar{b}' = \frac{b'_0 + b'_1 + b'_2}{3}
$$

这种技术通过“平均”参数，使模型的参数更加“通用”，从而提高泛化能力。

### 将模型权重保存到硬盘

在我们的代理中，我们已经将模型的最佳权重保存在 `Agent` 对象的 `best_weights` 属性中。这些权重仅保存在计算机的内存中。

如果我们有较长时间的训练，不想每次都重新训练，可以将模型权重保存到硬盘上，如下所示：

```python
def train(self, nb_epochs):
    for _ in range(nb_epochs):
        # 训练步骤省略...

        if val_loss <= np.min(self.val_losses):
            torch.save(self.model.state_dict(), f'model_weights_{self.name}.pth')
        else:
            print("⬈", end="")

def set_model_at_best(self):
    self.model.load_state_dict(torch.load(f'model_weights_{self.name}.pth'))
```

其中，`self.name` 是我们为代理添加的一个属性，用于命名我们的代理。

### 将整个模型保存到硬盘

以下技术可以保存一个包含所有参数和超参数（即整个架构）的模型。

```python
torch.save(model, 'model.pth')

# 加载模型
model = torch.load('model.pth')
```

### `pickle` 库

注意，`torch.save` 和 `torch.load` 基于 `pickle` 库，可以保存任何 Python 对象（非常方便）。

但要注意，pickle 基于“活代码”，这使得它对代码更新敏感。观察以下示例：

```python
import pickle

class Homme:
    def __init__(self, age, nom):
        self.age = age
        self.nom = nom

    def make_me_older(self):
        self.age += 1

    def print_me(self):
        print(f"Je m'appelle {self.nom} et j'ai {self.age} ans")

homme = Homme(12, "toto")
homme.make_me_older()
homme.print_me()

# 将对象保存到硬盘，包含所有属性
pickle.dump(homme, open("toto.pkl", "wb"))
```

现在，重启电脑。在 Colab 中：

```python
# 重启后重新定义类，但修改了一些方法
class Homme:
    def __init__(self, age, nom):
        self.age = age
        self.nom = nom

    def make_me_older(self):
        self.age += 1

    def print_me(self):
        print(f"my name is {self.nom} and I'm {self.age} year old.")

import pickle

homme_back = pickle.load(open("toto.pkl", "rb"))
homme_back.make_me_older()
homme_back.print_me()
```

**结果:** 当类定义发生变化时，pickle 加载时可能会出错或产生不一致的行为。

### 增加数据量

这可能是最好的技术。数据越多，训练数据越多，模型的泛化能力就越强。

但通常数据量有限。对于某些数据，如图片，可以通过引入旋转、平移、改变亮度等方法生成新数据。

**练习:** 调整 `keept` 参数，展示增加数据量如何减少过拟合。

```python
# 比较不同数据量下的训练效果

for keept in [2000, 4000, 8000, 16000]:
    print(f"\n数据量: {keept}")
    train_images_subset = train_images[:keept]
    train_labels_subset = train_labels[:keept]

    X_train_subset = np.reshape(train_images_subset / 255.0, [-1, 28*28])
    Y_train_subset = train_labels_subset

    X_train_torch_subset = torch.tensor(X_train_subset, dtype=torch.float32)
    Y_train_torch_subset = torch.tensor(Y_train_subset, dtype=torch.int64)

    model = Model_classif(hidden_dim)
    agent = Agent(model, learning_rate, X_train_torch_subset, Y_train_torch_subset, batch_size)
    agent.train(50)

    plt.plot(agent.losses, label='Training Loss')
    plt.plot(agent.val_steps, agent.val_losses, "r.", label='Validation Loss')
    plt.title(f'Data Size: {keept}')
    plt.legend()
    plt.show()
```

**解释:** 随着数据量的增加，模型不容易过拟合，验证损失下降更平稳，准确率更高。

### 减少参数数量

在深度学习中，模型中可训练参数的数量通常称为模型的“容量”。直观地，一个参数更多的模型具有更大的“记忆容量”，因此能够记住各种不必要的细节。

因此，我们需要在“容量过大”和“容量不足”之间找到平衡。

不幸的是，没有魔法公式可以确定合适的层数或每层的大小。通常有以下经验法则：

* 对于复杂数据，至少需要 3 层。
* 对于结构层次分明的数据，如图片，需要更多层（但这可能导致“梯度消失”问题）。

通常，我们通过逐步增加模型容量来寻找合适的架构。

**练习:** 通过调整 `hidden_dim` 参数，或者添加/删除层，来观察模型性能的变化。

```python
# 比较不同隐藏层维度的效果
for hidden_dim in [10, 50, 100, 200]:
    print(f"\n隐藏层维度: {hidden_dim}")
    model = Model_classif(hidden_dim)
    agent = Agent(model, learning_rate, X_train_torch, Y_train_torch, batch_size)
    agent.train(50)

    plt.plot(agent.losses, label='Training Loss')
    plt.plot(agent.val_steps, agent.val_losses, "r.", label='Validation Loss')
    plt.title(f'Hidden Dimension: {hidden_dim}')
    plt.legend()
    plt.show()
```

**解释:** 增加隐藏层维度通常会提高模型的学习能力，但也可能导致过拟合。需要根据数据量和验证结果调整。

### 惩罚项（正则化）

这是一种在训练过程中使模型性能稍差的技术，以使其对数据的调整程度较低，从而减少过拟合。我们将在下一个实验中详细介绍。

### 交叉验证

超参数（例如模型大小、提前停止的时机）的选择基于与训练集不同的验证集。

另一种方法是进行交叉验证：即使用“轮换”的验证集进行多次训练。例如，先使用第一部分作为验证集，然后使用第二部分，依此类推（每一部分称为一个“fold”）。

**优点 1:** 验证结果更加稳健。

**优点 2:** 可以了解模型的方差，即模型对验证集变化的敏感度。只需计算不同验证集结果的标准差即可。