amcl is a probabilistic localization system for a robot moving in 2D. It implements the adaptive \(or KLD-sampling\) Monte Carlo localization approach \(as described by Dieter Fox\), which uses a particle filter to track the pose of a robot against a known map.

amcl的英文全称是adaptive Monte Carlo localization，其实就是蒙特卡洛定位方法的一种升级版，使用自适应的KLD方法来更新粒子，这里不再多说（主要我也不熟），有兴趣的可以去看：KLD。

而mcl（蒙特卡洛定位）法使用的是粒子滤波的方法来进行定位的。而粒子滤波很粗浅的说就是一开始在地图空间很均匀的撒一把粒子，然后通过获取机器人的motion来移动粒子，比如机器人向前移动了一米，所有的粒子也就向前移动一米，不管现在这个粒子的位置对不对。使用每个粒子所处位置模拟一个传感器信息跟观察到的传感器信息（一般是激光）作对比，从而赋给每个粒子一个概率。之后根据生成的概率来重新生成粒子，概率越高的生成的概率越大。这样的迭代之后，所有的粒子会慢慢地收敛到一起，机器人的确切位置也就被推算出来了。![](https://img-blog.csdn.net/20151125153000523)

这幅图模拟了一个一维机器人的粒子更新，机器人下面那些想条形码一样的竖条就是粒子的分布了，可以看到粒子随着机器人的移动与更新会逐渐的收敛到机器人的正确位置上。

mcl算法步骤图：

![](https://img-blog.csdn.net/20151125153407039)

---

**参考链接：**

【1】[ros的navigation之———amcl（localization）应用详解](https://blog.csdn.net/chenxingwangzi/article/details/50038413)

