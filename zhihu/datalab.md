#! https://zhuanlan.zhihu.com/p/264941063
# CMU CS:APP3e学习-环境介绍&DataLab

>前提说明：我最近一直在学习CSAPP这门课，听两位教授讲课十分过瘾，也产生了一些心得。而在国庆期间无意看到了[@anarion](https://www.zhihu.com/people/anarion)大佬所写的学习笔记，深有感触，同时发现我和他的学习思路和使用工具(`jetbrains`,`vscode`,`linux`)都非常接近，于是也想把自己的心得和作业思路写成文章，分享在知乎上。
>我是一名普通的本科生，学这些课程只是出于自己的兴趣。我计划将这门课的学习写成一个专栏，以更加深入地理解课程内容，也算是对自己的一个监督。

[文档地址](https://github.com/rucnyz/blogs)

## 课程介绍

这门课程是卡内基梅隆大学的计算机基础课程，内容涵盖了计算机组成与体系结构，汇编，操作系统以及计算机网络等一些基础知识。相对应的那本深入理解计算机系统(csapp)的12个章节则对应着讲课的内容。建议大家可以去阅读这本书，内容十分详尽，就是一些特定的中文翻译可能不尽如人意。
我个人是以同时观看视频和阅读书籍的方式进行学习的。

## 课程资源

### 视频资源

[b站视频资源](https://www.bilibili.com/video/BV1iW411d7hd)
这位up主精校的字幕还不错，是对我这样英语不好的人的福音~。

### Lab资源

进入[CSAPP课程首页](http://csapp.cs.cmu.edu/3e/students.html)之后
![selfstudy](selfstudy.png)

- 点击红线，进入[Lab汇总](http://csapp.cs.cmu.edu/3e/labs.html)
- 后面两个为上课时的PPT以及代码示例，由于我没用到，因此不作介绍了。

进入之后，下图为第一个Lab：Data Lab的内容.
![self-study](datalab.png)
点击`README`可查看该作业的介绍；点击下载`Self-Study Handout`可获得源程序。

## 环境搭建

- [Ubuntu 20.04.1 LTS](https://cn.ubuntu.com/download)
- [CLion](https://www.jetbrains.com/clion/)
- CLion的Makefile support插件
![makefile](Makefile.png)

之后用`clion`打开`datalab-handout`，即可开始完成作业啦。

## DataLab及相关内容

此课程的第一部分是Representing and Manipulating Information，即信息的表示与处理。
由于本章视频内容不难，看完B站视频的前三节我就开始写datalab了。
但datalab真的很难，我花了好一段时间，和室友讨论了很久才算堪堪做完。
现在我将选取个人认为很困难的一些问题进行仔细分析。

**但首先的问题是怎么写这个lab。**

### 如何完成DataLab

![structure](first.png)
可以看到`datalab-handout`的内部文件结构，通过阅读README可以知道我们要填充的是`bits.c`(具体怎么填充请自行阅读),完成之后我们要做两件事。

1. 在clion自带的Terminal(当然也可以不用这个)中使用指令，没有任何反应则为正确

    ```shell
    ./dlc bits.c
    ```

2. 接着使用指令

    ```shell
    make btest
    ```

解释一下，`dlc`是一个专门的编译器，负责检查你所写的代码是否满足所规定的Legal ops和Max ops等规定，之后的`make btest`负责生成可执行文件`btest`，即检验你代码的正确性，包含你的每道题的错误和得分。
当然如果使用`clion`也可以进入`Makefile`文件直接点击左边的小三角运行。
![make btest](makebtest.png)

之后输入

```shell
./btest
```

即可查看得分。
另外，还有

```shell
make ishow
./ishow var1 var2...
make ftest
./fshow var1 var2...
```

用于帮助你理解和计算整数以及浮点数，比如

```shell
$ ./fshow 0x7F800000
Floating point value inf
Bit Representation 0x7f800000, sign = 0, exponent = 0xff, fraction = 0x000000
+Infinity

$ ./ishow 0xAAAAAAAA
Hex = 0xaaaaaaaa,       Signed = -1431655766,   Unsigned = 2863311530
```

README中还介绍了一些其他的指令用法，可自行探究。
下面我挑选一些为认为具有一定难度或是特点的问题进行分析。

### DataLab题目分析

>事先声明，以下代码为我本人所写，思路为本人所想或是和室友充分讨论后得出。
>部分注释借鉴了网络上的代码注释和思路，以更好地表达自己的意思。

#### bitXor

```C++
/*
 * bitXor - x^y using only ~ and &
 *   Example: bitXor(4, 5) = 1
 *   Legal ops: ~ &
 *   Max ops: 14
 *   Rating: 1
 */
int bitXor(int x, int y)
{
    return (~((~x) & (~y))) & (~(x & y));
}
```

第一道题很简单，但有意思的是我之后查其他人的做法时，发现大家都说与摩尔根定律有关，即

```C++
~(~x & ~y) == x | y
```

我并不太明白这里的摩尔根定律是指什么，是单纯的公式还是生物里的那个摩尔根？我没有查到资料。

无论如何，`x | y`和`~x | ~y`的并就是`x ^ y`，即可得到以上结果。

#### isTmax

```C++
/*
 * isTmax - returns 1 if x is the maximum, two's complement number,
 *     and 0 otherwise
 *   Legal ops: ! ~ & ^ | +
 *   Max ops: 10
 *   Rating: 1
 */
int isTmax(int x)
{
    return !(x + 1 + x + 1) & !!(x + 1);
}
```

这道题也很简单，但我开始没有想到可以用`x + x`的方式来表达`x << 1`，因此浪费了不少时间。

#### allOddBits

```C++
/*
 * allOddBits - return 1 if all odd-numbered bits in word set to 1
 *   where bits are numbered from 0 (least significant) to 31 (most significant)
 *   Examples allOddBits(0xFFFFFFFD) = 0, allOddBits(0xAAAAAAAA) = 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 2
 */
int allOddBits(int x)
{
    int tmp = 0xAA;
    tmp = (tmp << 8) + 0xAA;
    tmp = (tmp << 8) + 0xAA;
    tmp = (tmp << 8) + 0xAA;
    return !((tmp & x) ^ tmp);
}
```

这道题的思路并不复杂，核心在于用移位和加法来构造出`0xAAAAAAAA`，因为只有这个数是偶数位全为1，奇数位全为0，之后用任意的数和它做且运算，再用结果和它做异或运算即可分辨出两种数字。
由于数字大小限制，我使用了`0xAA`连续左移几次来达成这个目标。

#### isLessOrEqual

```C++
/*
 * isLessOrEqual - if x <= y  then return 1, else return 0
 *   Example: isLessOrEqual(4,5) = 1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 24
 *   Rating: 3
 */
int isLessOrEqual(int x, int y)
{
    //用于比较x和y的大小
    int a = ((x + ~y + 1) >> 31) & 0x1;
    int b = !(x ^ y);
    //用于处理溢出情况
    int c = (x >> 31) & 0x1;
    int d = (y >> 31) & 0x1;
    //结果
    return (a | b | (c & !d)) & !((!c) & d);
}
```

此题虽然结果比较复杂，但逻辑清晰明了。
首先易得

```C++
x + ~y + 1 == x - y;
```

当`x < y`时，有

```C++
(x + ~y + 1) >> 31 == 0xFFFFFFFF
0xFFFFFFFF & 0x1 == 0x1
!(x ^ y) == 0x0
0x1 | 0x0 == 0x1
```

当`x = y`时，有

```C++
(x + ~y + 1) >> 31 == 0x0
0x0 & 0x1 == 0x0
!(x ^ y) == 0x1
0x0|0x1 == 0x1
```

当`x > y`时，有

```C++
(x + ~y + 1) >> 31 == 0x0
0x0 & 0x1 == 0x0
!(x ^ y) == 0x0
0x0 | 0x0 == 0x0
```

这样就成功地把前两种情况和第三种情况分开。
c和d相关的操作考虑了两种溢出情况，当x为负数y为正数返回1，当x为正数y为负数返回0。

#### howManyBits

```C++
/* howManyBits - return the minimum number of bits required to represent x in
 *             two's complement
 *  Examples: howManyBits(12) = 5
 *            howManyBits(298) = 10
 *            howManyBits(-5) = 4
 *            howManyBits(0)  = 1
 *            howManyBits(-1) = 1
 *            howManyBits(0x80000000) = 32
 *  Legal ops: ! ~ & ^ | + << >>
 *  Max ops: 90
 *  Rating: 4
 */
int howManyBits(int x)
{
    int a, b, c, d, e, f;
    int sign = x >> 31; //得到0xFFFFFFFF或是0，用于处理负数
    x = (sign & ~x) | (~sign & x); //如果是正数将不发生变化，如果是负数将把所有位数全部变为1
    // 开始
    a = !!(x >> 16) << 4;//左边16位是否有1
    x = x >> a;//如果有，则将原数右移16位；若没有则不移动
    //第二轮
    b = !!(x >> 8) << 3;//不管左边16位是否有，此时只考虑留下的16位当中的左边8位
    x = x >> b;//如果有，则右移8位；若没有则不移动
    //第三轮
    c = !!(x >> 4) << 2;//之后同理
    x = x >> c;
    d = !!(x >> 2) << 1;
    x = x >> d;
    e = !!(x >> 1);
    x = x >> e;
    f = x;
    return a + b + c + d + e + f + 1;
}
```

这道题比较复杂，也不太好用语言表述。
基本的思路就是取16、8、4、2、1五种区域，然后
**1. 通过移位判断一段大区域当中是否有1；**
**2. 判断下一个可能有1的小区域中是否有1；**
**3. 执行操作1，直到区域已减为1。**
关于具体过程，我在注释中做了充分的解释。

#### floatScale2

```C++
/*
 * floatScale2 - Return bit-level equivalent of expression 2*f for
 *   floating point argument f.
 *   Both the argument and result are passed as unsigned int's, but
 *   they are to be interpreted as the bit-level representation of
 *   single-precision floating point values.
 *   When argument is NaN, return argument
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
unsigned floatScale2(unsigned uf)
{
    unsigned sign = uf & 0x80000000;//记录符号位，其余位置0
    unsigned exp = uf & 0x7F800000;//记录阶码，其余位置0
    unsigned frac = uf & 0x007FFFFF;//记录尾数，其余位置0
    //如果uf的阶码为0，即非规格化值
    if (!exp)
    {
        //将frac左移一位，若尾数部分第一位为0，尾数左移一位就相当于乘2，仍然是非规格化数;
        //若尾数部分第一位为1，左移前为非规格化数，左移后阶码部分由00000000变为
        //00000001，阶码由-126变为1-127=-126没有任何变化,尾数以二进制小数解释的方式将变为(1+frac)*2
        //而变为规格化数后尾数被解释为1+frac，最终就是尾数*2，阶码不变。
        frac = frac << 1;
    }
    //如果阶码部分不为全1，即规格化值
    else if (exp ^ 0x7F800000)
    {
        exp += 0x00800000;//阶码加所在位置的'1'，对于规格化数，相当于*2
        //如果加1后，阶码全为1，则溢出；令尾数等于0，使得返回值为无穷大
        if (!(exp ^ 0x7F800000))
        {
            frac = 0;
        }
    }
    //对于阶码为在本身为全1的NaN，将会返回原数据，满足此题要求
    return sign | exp | frac;//对符号位，阶码位，尾数位进行异或运算
}
```

这道题不难，实际上如果熟悉浮点数编码，应当能够很快写出来。
这道题重要的一点在于**让人理解了为什么要把非规格化值的阶码值设置为1-Bias**
关于这个我在注释中做了仔细的解释。

#### floatFloat2Int

```C++
/*
 * floatFloat2Int - Return bit-level equivalent of expression (int) f
 *   for floating point argument f.
 *   Argument is passed as unsigned int, but
 *   it is to be interpreted as the bit-level representation of a
 *   single-precision floating point value.
 *   Anything out of range (including NaN and infinity) should return
 *   0x80000000u.
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
int floatFloat2Int(unsigned uf)
{
    int sign = uf & 0x80000000;//记录符号位，其余位置0
    int exp = uf & 0x7F800000;//记录阶码，其余位置0
    int frac = uf & 0x007FFFFF;//记录尾数，其余位置0
    //全部移到右边方便计算
    sign = sign >> 31;
    exp = (exp >> 23) - 127;
    //只要超过2^31即溢出
    if (exp > 31)
    {
        return 0x80000000u;
    }
    //exp小于0,则原数<1,返回0即可
    if (exp < 0)
    {
        return 0;
    }
    //frac加上默认的1
    frac += (1 << 23);
    //<=23，右移
    if (exp <= 23)
    {
        frac = frac >> (23 - exp);
    }
        //>=24，左移
    else if (exp <= 31)
    {
        frac = frac << (exp - 24);
    }
    //如果符号位为1，转负数
    if (sign)
    {
        frac = ~frac + 1;
    }
    return frac;
}
```

此题同样不复杂，另外不需要考虑非规格化数的情况，因此我们只需要对exp进行几个分段就可以完成了。
后面根据阶码的大小对frac进行移动需要一番思考，但通过看书和视频应当也不是很困难。

#### 其他题目的答案

```C++
/*
 * tmin - return minimum two's complement integer
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 4
 *   Rating: 1
 */
int tmin(void)
{
    int a = 0x1;
    return a << 31;
}
/*
 * negate - return -x
 *   Example: negate(1) = -1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 5
 *   Rating: 2
 */
int negate(int x)
{
    return ~x + 1;
}
/*
 * isAsciiDigit - return 1 if 0x30 <= x <= 0x39 (ASCII codes for characters '0' to '9')
 *   Example: isAsciiDigit(0x35) = 1.
 *            isAsciiDigit(0x3a) = 0.
 *            isAsciiDigit(0x05) = 0.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 15
 *   Rating: 3
 */
int isAsciiDigit(int x)
{
    int a = 0x30;
    int b = 0x3A;
    return (!((x + (~a + 1)) >> 31)) & ((x + (~b + 1)) >> 31);
}
/*
 * conditional - same as x ? y : z
 *   Example: conditional(2,4,5) = 4
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 16
 *   Rating: 3
 */
int conditional(int x, int y, int z)
{
    int a = !x + ~0x1 + 0x1;
    return (a & y) | (~a & z);
}
/*
 * logicalNeg - implement the ! operator, using all of
 *              the legal operators except !
 *   Examples: logicalNeg(3) = 0, logicalNeg(0) = 1
 *   Legal ops: ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 4
 */
int logicalNeg(int x)
{
    return ((~x & ~(~x + 1)) >> 31) & 0x1;
}
/*
 * floatPower2 - Return bit-level equivalent of the expression 2.0^x
 *   (2.0 raised to the power x) for any 32-bit integer x.
 *
 *   The unsigned value that is returned should have the identical bit
 *   representation as the single-precision floating-point number 2.0^x.
 *   If the result is too small to be represented as a denorm, return
 *   0. If too large, return +INF.
 *
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. Also if, while
 *   Max ops: 30
 *   Rating: 4
 */
unsigned floatPower2(int x)
{
    //溢出
    if (x > 128)
    {
        return 0x7F800000;
    }
    //太小
    if (x < -126)
    {
        return 0;
    }
    //正常情况，加上偏移127
    return (x + 127) << 23;
}
```

个人感觉，datalab中的难题主要在整数一块；float的题目虽然复杂，但思路都比较常规，按部就班即可完成，不像是前面的一些题目比如howManyBits着实需要灵光乍现。

之后我将继续学习后面的内容，下一次的作业将是BombLab，是一个很有意思的'拆炸弹'题，我将在之后发布。
