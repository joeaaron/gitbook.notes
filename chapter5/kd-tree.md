KD树的全称为k-Dimension Tree的简称，是一种分割K维空间的数据结构，主要应用于关键信息的搜索。为什么说是K维的呢，因为这时候的空间不仅仅是2维度的，他可能是3维，4维度的或者是更多。我们举个例子，如果是二维的空间，对于其中的空间进行分割的就是一条条的分割线，比如说下面这个样子。

![](https://img-blog.csdn.net/20150410214415026)

如果是3维的呢，那么分割的媒介就是一个平面了，下面是3维空间的分割

![](https://img-blog.csdn.net/20150410214612600)

这就稍稍有点抽象了，如果是3维以上，我们把这样的分割媒介可以统统叫做超平面 。那么KD树算法有什么特别之处呢，还有他与K-NN算法之间又有什么关系呢，这将是下面所将要描述的。

**KNN**

KNN就是K最近邻算法，他是一个分类算法，因为算法简单，分类效果也还不错，也被许多人使用着，算法的原理就是选出与给定数据最近的k个数据，然后根据k个数据中占比最多的分类作为测试数据的最终分类。图示如下：

![](https://img-blog.csdn.net/20150410215111995)

算法固然简单，但是其中通过逐个去比较的办法求得最近的k个数据点，效率太低，时间复杂度会随着训练数据数量的增多而线性增长。于是就需要一种更加高效快速的办法来找到所给查询点的最近邻，而KD树就是其中的一种行之有效的办法。但是不管是KNN算法还是KD树算法，他们都属于相似性查询中的K近邻查询的范畴。在相似性查询算法中还有一类查询是范围查询，就是给定距离阈值和查询点，dbscan算法可以说是一种范围查询，基于给定点进行局部密度范围的搜索。想要了解KNN算法或者是Dbscan算法的可以点击我的K-最近邻算法和Dbscan基于密度的聚类算法。

KD-Tree

在KNN算法中，针对查询点数据的查找采用的是线性扫描的方法，说白了就是暴力比较，KD树在这方面用了二分划分的思想，将数据进行逐层空间上的划分，大大的提高了查询的速度，可以理解为一个变形的二分搜索时间，只不过这个适用到了多维空间的层次上。下面是二维空间的情况下，数据的划分结果：

现在看到的图在逻辑上的意思就是一棵完整的二叉树，虚线上的点是叶子节点。

KD树的算法原理

KD树的算法的实现原理并不是那么好理解，主要分为树的构建和基于KD树进行最近邻的查询2个过程，后者比前者更加复杂。当然，要想实现最近点的查询，首先我们得先理解KD树的构建过程。下面是KD树节点的定义，摘自百度百科：

域名

数据类型

描述

Node-data

数据矢量

数据集中某个数据点，是n维矢量（这里也就是k维）

Range

空间矢量

该节点所代表的空间范围

split

整数

垂直于分割超平面的方向轴序号

Left

k-d树

由位于该节点分割超平面左子空间内所有数据点所构成的k-d树

Right

k-d树

由位于该节点分割超平面右子空间内所有数据点所构成的k-d树

parent

k-d树

父节点

变量还是有点多的，节点中有孩子节点和父亲节点，所以必然会用到递归。KD树的构建算法过程如下\(这里假设构建的是2维KD树，简单易懂，后续同上\)：

1、首先将数据节点坐标中的X坐标和Y坐标进行方差计算，选出其中方差大的，作为分割线的方向，就是接下来将要创建点的split值。

2、将上面的数据点按照分割方向的维度进行排序，选出其中的中位数的点作为数据矢量，就是要分割的分割点。

3、同时进行空间矢量的再次划分，要在父亲节点的空间范围内再进行子分割，就是Range变量，不理解的话，可以阅读我的代码加以理解。

4、对剩余的节点进行左侧空间和右侧空间的分割，进行左孩子和右孩子节点的分割。

5、分割的终点是最终只剩下1个数据点或一侧没有数据点的情况。

在这里举个例子，给定6个数据点：

（2,3），（5,4），（9,6），（4,7），（8,1），（7,2）

对这6个数据点进行最终的KD树的构建效果图如下，左边是实际分割效果，右边是所构成的KD树：

x，y代表的是当前节点的分割方向。读者可以进行手动计算并验证，本人不再加以描述。

KD树构建完毕，之后就是对于给定查询点数据，进行此空间数据的最近数据点，大致过程如下：

1、从根节点开始，从上往下，根据分割方向，在对应维度的坐标点上，进行树的顺序查找，比如给定\(3,1\)，首先来到\(7,2\),因为根节点的划分方向为X，因此只比较X坐标的划分，因为3&lt;7，所以往左边走，后续的节点同样的道理，最终到达叶子节点为止。

2、当然以这种方式找到的点并不一定是最近的，也许在父节点的另外一个空间内存在更近的点呢，或者说另外一种情况，当前的叶子节点的父亲节点比叶子节点离查询点更近呢，这也是有可能的。

3、所以这个过程会有回溯的步骤，回溯到父节点时候，需要做2点，第一要和父节点比，谁里查询点更近，如果父节点更近，则更改当前找到的最近点，第二以查询点为圆心，当前查询点与最近点的距离为半径画个圆，判断是否与父节点的分割线是否相交，如果相交，则说明有存在父节点另外的孩子空间存在于查询距离更短的点，然后进行父节点空间的又一次深度优先遍历。在局部的遍历查找完毕，在于当前的最近点做比较，比较完之后，继续往上回溯。

下面给出基于上面例子的2个测试例子，查询点为（2.1，3.1）和\(2，4.5\)，前者的例子用于理解一般过程，后面的测试点真正诠释了递归，回溯的过程。先看下\(2.1，3.1\)的情况：

因为没有碰到任何的父节点分割边界，所以就一直回溯到根节点，最近的节点就是叶子节点\(2，3\).下面\(2,4.5\)是需要重点理解的例子，中间出现了一次回溯，和一次再搜索：

在第一次回溯的时候，发现与y=4碰撞到了，进行了又一次的搜寻，结果发现存在更近的点，因此结果变化了，具体的过程可以详细查看百度百科-kd树对这个例子的描述。
