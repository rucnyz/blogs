# pytorch: Tensor基础函数 

##  1. Tensor的数据类型

在PyTorch中，主要有10种类型的tensor，其中重点使用的为以下八种(还有BoolTensor和BFloat16Tensor)：
| Data type | dtype | dtype |
| :---: | :---: | :---: |
| 32-bit floating point | torch.float32 or torch.float | torch.(cuda).FloatTensor |
| 64-bit floating point | torch.float64 or torch.double | torch.(cuda).DoubleTensor |
| 16-bit floating point | torch.float16 or torch.half | torch.(cuda).HalfTensor |
| 8-bit integer(unsigned) | torch.uint8 | torch.(cuda).ByteTensor |
| 8-bit integer(signed) | torch.int8 | torch. (cuda). CharTensor |
| 16-bit integer(signed) | torch.int16 or torch.short | torch.(cuda).ShortTensor |
| 32-bit integer(signed) | torch.int32 or torch.int | torch.(cuda).IntTensor |
| 64-bit integer(signed) | torch.int64 or torch.long | torch.(cuda).LongTensor |

在具体使用时可以根据网络模型所需的精度和显存容量进行选取。

* 一般情况而言，模型参数和训练数据都是采用默认的32位浮点型；16位半精度浮点是为在GPU上运行的模型所设计的，因为这样可以尽可能节省GPU的显存占用。
当然这样节省显存空间也会缩小所能表达数据的能力。因此自pytorch1.6自动混合精度(automatic mixed precision)被更新后，将半精度和单精度混合使用以达到减少显存、提升推理速度的方法就被大面积的推广开来。在这里不对自动混合精度(AMP)模块做过多介绍。

* 训练用的标签一般是整型中的32或64位，而在硬盘中存储数据集文件时则一般采用uint8来存分类标签(除非超过了255类)，总之就是尽可能在不损失信息的情况下压缩空间。

对于tensor之间的类型转换，可以通过`type()`,`type_as()`,`int()`等进行操作。其中不传入`dtype`参数的`type()`函数为查看当前类型，传入参数则是转换为该参数代表的类型。

```python
# 创建新的Tensor时默认类型为float32位
>>> a = torch.randn(2, 2)
>>> a.type()
'torch.FloatTensor'

# 使用int()、float()、double()等进行数据转换
>>> b = a.double()
>>> b
tensor([[ 0.1975, -0.3009],
        [ 1.7323, -0.4336]], dtype=torch.float64)

# 使用传入dtype参数的type()函数
>>> a.type(torch.IntTensor)        
tensor([[0, 0],
        [1, 0]], dtype=torch.int32)

# 使用type_as()函数,将a的类型转换为b的类型
>>> a.type_as(b)
tensor([[ 0.1975, -0.3009],
        [ 1.7323, -0.4336]], dtype=torch.float64)
```

注意这里提到默认类型为`float32`，但是在使用`from_numpy()`函数时创建的`tensor`将会和原本的`ndarray`的类型保持一致，这个问题将在下一节具体讨论。

值得一提的是`type_as()`函数非常方便，在实际建立模型的过程中经常需要保持tensor之间的类型一致，我们只需要使用`type_as()`即可，不需要操心具体是什么类型。

##  2. Tensor的创建与查看

`Tensor`有很多创建方式，最常用基本的就是`tensor()`，而构造函数`Tensor()`用的很少。同时也有很多和`numpy`类似的创建方式，比如`ones()`、`zeors()`、`randn()`等等。
接下来我用代码的方式来介绍常见的创建方式，以及一些容易混淆的情况。

本节涉及函数

* [`torch.tensor()`、`torch.Tensor()`、`torch.DoubleTensor()`](#21-基础tensor函数)
* [`torch.ones()`、`torch.zeros()`、`torch.eye()`](#221-特殊矩阵创建方法)
* [`torch.randn()`、`torch.randperm()`、``torch.randint()``、`torch.rand()`](#222-随机矩阵创建方法)
* [`torch.ones_like()`](#223-like方法)
* [`torch.arange()`、`torch.linespace()`、`torch.logspace()`](#224-序列创建方法)

###  2.1. 基础tensor函数

`tensor()`是最常使用的，从`data`参数来构造一个新的tensor，下为官方文档的介绍
>**data** (array_like) – Initial data for the tensor. Can be a list, tuple, NumPy `ndarray`, scalar, and other types.
基本上任何矩阵模样的数据都可通过`tensor()`被转换为`tensor`

`Tensor()`是最原始的构造函数，不建议使用

```python
# Tensor()参数为每一维大小，得到数据为随机初始化
>>> torch.Tensor(2,2)
tensor([[0.0000, 4.4766],
        [0.0000, 0.0000]])

# tensor()的常见用法和特殊情况
>>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor([[ 0.1000,  1.2000],
        [ 2.2000,  3.1000],
        [ 4.9000,  5.2000]])

>>> torch.tensor([0, 1])  # 会自行类型推断，创建int类型
tensor([ 0,  1])

>>> torch.tensor([])  # 创建一个空tensor (size (0,))
tensor([])
```

使用Tensor内置的各种数据类型进行创建

```python
torch.DoubleTensor(2, 2)
```

###  2.2. 类numpy方法

####  2.2.1. 特殊矩阵创建方法

```python
>>> torch.zeros(2,2)# 全为1的矩阵
>>> torch.ones(2,2) # 全为0的矩阵
>>> torch.eye(2,2) # 单位矩阵
```

####  2.2.2. 随机矩阵创建方法

```python
# 按照所给维度创建矩阵，用标准正态分布(N(0,1))中的随机数来填充
>>> torch.randn(2,2)
tensor([[ 0.9798,  0.4567],
        [-0.4731, -0.3492]])

# 和randn一样，但是这次是用[0,1]均匀分布中的随机数来填充
>>> torch.rand(2,2)
tensor([[0.4497, 0.3038],
        [0.1431, 0.0814]])

# 生成长度为n的随机排列向量
>>> torch.randperm(5)
tensor([0, 4, 1, 3, 2])

# 用0-n的随机整数来填充矩阵，第二个参数为要求的维度
# 此为[0-4]的整数
>>> torch.randint(4, (2,3))
tensor([[2, 0, 3],
        [0, 0, 1]])
```

####  2.2.3. like方法

按照所给tensor维度生成相同维度的目标矩阵，这里就只举一个例子好了

```python
>>> a = torch.randn(2, 3)
>>> torch.ones_like(a)
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

####  2.2.4. 序列创建方法

按照所给区间创建各种序列
>注：`range()`函数是deprecated状态，不在此介绍

`arange()`函数区间为[start, end)

```python
# 只传入一个参数end，根据[0, end)区间中创建序列
>>> torch.arange(5)
tensor([0, 1, 2, 3, 4])
# 传入两个参数start和end，根据[start, end)区间中创建序列
>>> torch.arange(1, 5)
tensor([1, 2, 3, 4])
# 传入step参数，代表间隔
>>> torch.arange(1, 8, step = 2)
tensor([1, 3, 5, 7])
```

linespace区间为[start,end]，但此处steps参数代表生成的tensor中元素数量
在区间中根据生成数量进行线性插值返回tensor，返回tensor元素为：
$$
\left(\text { start, start }+\frac{\text { end }-\text { start }}{\text { steps }-1}, \ldots, \text { start }+(\text { steps }-2) * \frac{\text { end }-\text { start }}{\text { steps }-1},\right. \text { end) }
$$

```python
>>> torch.linspace(3, 10, steps=5)
tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
>>> torch.linspace(-10, 10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
# 若steps为1，相当于间隔无限大，就只得到一个元素的tensor
>>> torch.linspace(start=-10, end=10, steps=1)
tensor([-10.])
```

logspace基本和linespace一致，不过使用的是指数函数进行插值
$$
\text { (base } \left.^{\text {start }}, \text { base }^{\text {(start+ } \left.\frac{\text { end - start }}{\text { steps }-1}\right)}, \ldots, \text { base }^{\text {(start } \left.+\text { (steps }-2) * \frac{\text { end - start }}{\text { steps }-1}\right)}, \text { base }^{\text {end }}\right)
$$

```python
# base默认为10
>>> torch.logspace(start=-10, end=10, steps=5)
tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
>>> torch.logspace(start=0.1, end=1.0, steps=1)
tensor([1.2589])
# 可以传入base参数改变基底
>>> torch.logspace(start=2, end=2, steps=1, base=2)
tensor([4.0])
```

##  3. Tensor的组合与分块

本节涉及函数

* [`torch.cat(),torch.stack()`](#31-组合操作)
* [`torch.chunk(), torch.split()`](#32-分块操作)

###  3.1. 组合操作

组合是指将不同的Tensor叠加起来，主要有`torch.cat()`和`torch.stack()`两个函数。

`cat()`(或`concat()`)是concatenate的意思，即沿着已有的数据的某一维度进行拼接，操作后数据的总维数不变，在进行拼接时，**除了拼接的维度之外，其他维度必须相同**
  
```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])

# 按照第0个维度进行拼接
>>> y = torch.cat((x, x, x), 0) 
>>> y.shape
torch.Size([6, 3])

# 按照第1个维度进行拼接
>>> torch.cat((x, x, x), 1).shape 
torch.Size([2, 9])

# 用于拼接的维度可以不相等
>>> torch.cat((x, y), 0).shape 
torch.Size([8, 3])
```

`stack()`函数则是指新增维度，并按照指定的维度进行叠加，**所有tensor的维度必须完全相同**

```python
>>> x = torch.randn(3, 4)

# 以第0维进行stack，输出维度为2x3x4,效果就是叠加序列本身
>>> torch.stack((x, x), 0).shape
torch.Size([2, 3, 4])

# 以第1维进行stack，输出维度为3x2x4，效果是按照每一行叠加
>>> torch.stack((x, x), 1).shape
torch.Size([3, 2, 4])

# 以第2维进行stack，输出维度为3x4x2，效果是按每一行的每一个元素进行叠加
>>> torch.stack((x, x), 2).shape
torch.Size([3, 4, 2])
```

###  3.2. 分块操作

分块是与组合相反的操作，指将`Tensor`分割成不同的子`Tensor`，主要有`torch.chunk()`和`torch.split()`两个函数，前者需要指定分块的数量而后者则需要指定每一块的大小。

`chunk()`需要指定**分的块数**和按照哪一维分块

```python
>>> a = torch.arange(10) # size(10,)

# 可以除尽，返回五个tensor，每个tensor维度为2
>>> a.chunk(5)
(tensor([0, 1]),
 tensor([2, 3]),
 tensor([4, 5]),
 tensor([6, 7]),
 tensor([8, 9]))

# 无法除尽，则保证前面维数一致，最后一个tensor不一样
>>> a.chunk(4)
(tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6, 7, 8]), tensor([9]))

# 还有可能无法除尽同时也无法得到要求数量的tensor，那么返回tensor数量会减少
>>> torch.arange(11).chunk(5)
(tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6, 7, 8]), tensor([ 9, 10]))

# 按照其他维度分块的例子
# 按照第二个维度分成两块
>>> torch.randn(3, 4).chunk(2,1)
(tensor([[ 0.7786,  0.6219],
         [-0.1352, -0.3261],
         [-0.9451, -1.1154]]),
 tensor([[1.3665, 0.8111],
         [0.8320, 1.9941],
         [0.9997, 0.6056]]))
# 即得到两个3x2的tensor
```

`split()`函数需要指定**每一块的大小**和按照哪一维分块

```python
# 沿着第0维分块，每一块维度为2，所以就是没分
>>> torch.randn(2,3).split(2,0)
(tensor([[ 0.2060,  1.0265, -1.0841],
         [ 1.2017,  0.1215,  0.7324]]),)

# 沿着第1维分块，要求每一块维度为2，无法除尽，所以第一个tensor为2x2，第二个tensor为2x1
>>> torch.randn(2,3).split(2,1)
(tensor([[ 0.0727,  0.4330],
         [-0.0220,  1.6440]]),
 tensor([[-0.0685],
         [ 0.3101]]))
```

##  4. Tensor的索引

涉及函数

* [`index_select()`、`masked_select()`](#41-下标索引)
* [`where()`、`clamp()`](#42-选择索引)

索引操作与Numpy非常类似，主要包括下标索引、表达式索引和选择索引

###  4.1. 下标索引

```python
>>> a = torch.randn(2,3)

# 根据下标进行索引，用函数表达是index_select()
>>> a[1]
tensor([ 1.0374,  1.1266, -1.8777])
>>> a[1, 2]
tensor(-1.8777)

# index_select需要传进去dim和indices两个参数
>>>a.index_select(0,torch.tensor([0]))

# 选择符合条件的元素并返回，用函数表达是masked_select()
>>> a[a>0]
tensor([0.0258, 1.0374, 1.1266])
>>> a.masked_select(a>0)
tensor([0.0258, 1.0374, 1.1266])
```

###  4.2. 选择索引

```python
# self.where()需要传入condition和other参数，即将other的数据填充到condition中False的地方
>>> a
tensor([[-1.0293, -2.0182,  0.0258],
        [ 1.0374,  1.1266, -1.8777]])
>>> a.where(a > 0, torch.ones(1, 3))
tensor([[1.0000, 1.0000, 0.0258],
        [1.0374, 1.1266, 1.0000]])

# 使用torch.where(condition, a, y)函数相当于a.where(condition, y)
```

`where()`非常值得注意的一点是，用于填充的other参数遵守torch的广播机制，并且需要保证最后一个维度和原tensor的最后一维保持一致或者为1

也就是说在上方代码的情况中，torch.ones的维度可以是(1)、(3)、(1,3)、(1,1)、(2,3)，但不能是(2)

```python
# 对Tensor元素进行限制可以使用clamp()函数
>>> a.clamp(1, 2) # 将不在[1,2]范围内的元素放大到1或者缩小到2
tensor([[1.0000, 1.0000, 1.0000],
        [1.0374, 1.1266, 1.0000]])
```

##  5. Tensor的维度变形

维度转换指改变`Tensor`的维度，以适应在深度学习的计算中，数据维度经常变换的需求，在pytorch中主要有四类不同的变形方法

* [`view()`、`resize()`、`reshape()`](#51-调整形状)
* [`transpose()`、`permute()`](#52-维度之间的转换)
* [`squeeze()`、`unsqueeze()`](#53-处理size为1的维度)
* [`expand()`、`expand_as()`](#54-复制元素来扩展维度)

下面将按照类别介绍它们

###  5.1. 调整形状

`view()`、`resize()`、`reshape()`函数可以在不改变`Tensor`数据的情况下任意改变Tensor的形状，调整前后共享内存，三者作用基本相同

>`resize()`现在已经处于deprecated状态，只保留了进行in-place操作的`resize_()`

```python
>>> a = torch.arange(0,4)
>>> a
tensor([0, 1, 2, 3])

>>> b = a.view(2, 2)
>>> b
tensor([[0, 1],
        [2, 3]])

>>> c = a.reshape(4, 1) # resize一样
>>> c
tensor([[0],
        [1],
        [2],
        [3]])
>>> c[1, 0]=0
>>> b
tensor([[0, 0],
        [2, 3]])
```

如果想要直接改变`Tensor`的尺寸，可以使用`resize_()`原地操作。在`resize_()`函数中，如果超过了原`Tensor`的大小则重新分配内存，多出部分置零，如果小于原`Tensor`大小则剩余的部分仍然会隐藏保留

###  5.2. 维度之间的转换

`transpose()`函数可以将指定的两个维度的元素进行转置，而`permute()`函数则可以按照给定的维度进行维度变换

```python
>>> a=torch.randn(2,3,1)
>>> a
tensor([[[-1.1151],
         [ 2.6100],
         [-0.0333]],
        [[ 0.6966],
         [ 0.3621],
         [-0.7940]]])


# 将第0维和第1维的元素进行转置,且维度变为(3,2,1)
>>> a.transpose(0, 1)
tensor([[[-1.1151],
         [ 0.6966]],
        [[ 2.6100],
         [ 0.3621]],
        [[-0.0333],
         [-0.7940]]])


# 按照第2、1、0的维度顺序重新进行元素排列，维度变为(1,3,2)
>>> a.permute(2, 1, 0)
tensor([[[-1.1151,  0.6966],
         [ 2.6100,  0.3621],
         [-0.0333, -0.7940]]])
```

###  5.3. 处理size为1的维度

在实际的应用中，经常需要增加或减少`Tensor`的维度，尤其是维度为1的情况(特别是处理label的时候，经常有(n,1)和(n,)相互转换的需求)。
`squeeze()`用于去除size为1的维度，而`unsqueeze()`用于将指定的维度size变为1

```python
>>> a=torch.arange(0,4)
>>> a.shape
torch.Size([4])

# 将第0维变为1，因此总维度为(1,4)
>>> b = a.unsqueeze(0)
>>> b.shape
torch.Size([1, 4])

# 第0维如果是1，则去掉该维度，如果不是则不进行任何操作
>>> b.squeeze(0).shape
torch.Size([4])
>>> b.squeeze(1).shape
torch.Size([1, 4])
```

###  5.4. 复制元素来扩展维度

有时需要采用复制元素的方式来扩展`Tensor`的维度，这时`expand`相关函数就派上用场了。`expand()`函数将`size`为1的维度复制扩展为指定大小，也可以使用`expand_as()`函数指定为目标`Tensor`的维度

```python
>>> a=torch.randn(2,1)
>>> a
tensor([[-0.0026],
        [-0.8002]])

# 将第1维的维度由1变为3，则复制该维的元素，并扩展为3
>>> a.expand(2,3)
tensor([[-0.0026, -0.0026, -0.0026],
        [-0.8002, -0.8002, -0.8002]])
```

>在进行`Tensor`操作时，有些操作比如`transpose()`、`permute()`可能会将`Tensor`在内存中变得不连续，而有些操作如`view()`是需要内存连续的，这种情况下可以采用`contiguous()`操作先将内存变为连续，而`reshape()`操作相当于帮助我们整合了这个情况，也就是`self.reshape()`函数等于`self.contiguous().view()`。
而`resize()`函数没有了自己独特的功能，如今已经被deprecated了

##  6. Tensor的内存共享

涉及函数

* [`from_numpy()`、`numpy()`](#62-tensor与numpy转换)
* [`as_tensor()`、`tensor()`、`tolist()`](#63-tensor与其他类型数据转换)

### 6.1. 通过Tensor初始化Tensor

直接通过Tensor来初始化另一个Tensor，或是通过Tensor的组合、分块、索引、变形操作来初始化另一个Tensor，则这两个Tensor共享内存

```python
>>> a=torch.randn(2, 2)
>>> a
tensor([[0.1783, 0.5609],
        [0.8006, 0.8315]])
# 用a初始化b，或者用a的变形操作初始化c，则这三者共享内存
>>> b=a
>>> c=a.view(4)
>>> b[0,0]=0
>>> c[3]=4
>>> a
tensor([[0.0000, 0.5609],
        [0.8006, 4.0000]])

```

### 6.2. Tensor与Numpy转换

```python
>>> a=torch.randn(2,2)
>>> a
tensor([[-0.0560,  0.5018],
        [ 0.1275, -1.0663]])

# 转为numpy
>>> b = a.numpy()
>>> b
array([[-0.05598828,  0.50184375],
       [ 0.12752114, -1.0663038 ]], dtype=float32)
# 注意这里默认会保持相同的数据类型，也就是转换为单精度的float32

# numpy转为Tensor
>>> c = torch.from_numpy(b)
>>> c
tensor([[-0.0560,  0.5018],
        [ 0.1275, -1.0663]])
>>> torch.from_numpy()
```

在实际进行深度学习的过程中，总是会有双精度和单精度转换的问题，而`from_numpy()`函数会**保留原`ndarray`的数据类型**。

在pytorch训练中默认采用`FloatTensor`，所以如果其他来源的`ndarray`是`float64`的类型，那么`from_numpy()`转换过去将会变为pytorch中的`DoubleTensor`类型，从而导致数据类型不匹配

这时候就需要再使用`float()`函数，即`torch.from_numpy().float()`，来将双精度转换为单精度数据。

### 6.3. Tensor与其他类型数据转换

```python
>>> a = np.array([1, 2, 3])
>>> t = torch.as_tensor(a) # 内存共享
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([-1,  2,  3])

>>> b = t.tolist() # 内存不共享
>>> t[0] = 1
>>> b
[-1, 2, 3]
```

最后我提供一个表格包含了上述的数据类型转换以及是否内存共享
| 函数 | 简述 | 是否和原数据内存共享 |
| :---: | :---: | :---: |
| torch.tensor() | 从任何类数组创建tensor | 否 |
| torch.as_tensor() | 从任何类数组创建tensor | 是 |
| torch.from_numpy() | 从numpy创建tensor | 是 |
| torch.numpy() | 取出tensor内部存储的numpy数组 | 是 |
| torch.tolist() | 将tensor转换为list | 否 |

注意这里使用的函数，均会和原数据的数据类型保持一致，所以`float()`应当会是一个相当常用的适配函数。

## Reference

[pytorch官方文档(1.10.0版)](https://pytorch.org/docs/stable/index.html)

[Object Detection by Deep Learning：Core Technologie](https://github.com/dongdonghy/Detection-PyTorch-Notebook)