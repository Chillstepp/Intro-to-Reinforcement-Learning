---
title: 强化学习纲要Ch5-无模型价值函数估计和控制-下
date: 2021-05-20 21:59:15
index_img: /img/rl_zbl.png
tags: 强化学习
---

# 无模型的价值函数估计和控制—下

上一节讲了预测(prediction)问题，这一节我们来解决控制(control)问题。

![image-20210506222813211](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210506222813211.png)

我们之前再policy evaluation中用的方法是动态规划迭代，而上一节提到了一种新的做法也就是通过MC方法来做在特定策略$\pi$下计算状态的价值函数。

**model-free时control问题的解决方法：**

- 在model-free中，我们首先要用MC方法来填一个表格Q-table，即policy evaluation：

![image-20210506224528600](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210506224528600.png)

- 然后去更新策略，即control的第二步策略提升policy improvement



在MC算Q-table时有一个trade-off，也就是exploration和exploitation之间的trade-off。

![image-20210506225241469](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210506225241469.png)

**$\epsilon$-Greedy Exploration​ ：**

提出了在策略$\pi$下，一个状态在策略$\pi$下不仅可以对应着一个当前收益最大的act，还会有一个随机的act，这个随机的act就是为了exploration。

这个$\epsilon$是可以变化的，在前期可以大一些，后面可以逐渐变小。



MC with $\epsilon$-Greedy Exploration算法有一个特点：在新策略下的价值函数总比以前就策略下的价值函数大。下面给出一个简单的证明：

![image-20210506230233190](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210506230233190.png)



![image-20210506234705288](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210506234705288.png)



**Sarsa算法（On-Policy TD Control）：**

![image-20210511212737480](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210511212737480.png)

Sarsa这名字的来历就是：SARSA分开来看，根据当前状态，做出ACT，得到Reward，然后转移到一个状态，做出新的ACT。

伪代码：

![image-20210511214856933](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210511214856933.png)



即然存在n-step TD，那么也会存在n-step sarsa。





**On-policy  vs.  Off-policy Learning：**

On-policy是 学习策略$\pi$通过策略$\pi$所产生的轨迹数据。

Off-policy是指在学习策略$\pi$时，用了两种策略产生轨迹数据给他学习：一种策略是现在学到的策略，也是我们希望最优化的策略，我们一般称之为目标策略target policy $\pi$。另一个 策略是我们拿来探索的策略，那么这个探索的策略可以激进一些，我们称之为行为策略behavior policy $\mu$ ，这个行为策略通过探索找到一些新的轨迹数据，然后再喂给target policy来学习。

on-policy 与 off-policy的本质区别在于：更新Q值时所使用的方法是沿用既定的策略（on-policy）还是使用 新策略（off-policy）。



**Off-Policy Control with Q-Learning：**

![image-20210511220835055](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210511220835055.png)

Q-Learning 就是一种Off-Policy的算法，一种target policy是按照贪心的方法选择当前已知最好的方法去更新，另一种就是behavior policy，但是这里她并没有采用完全的随机，因为完全的随机其实一般效果不会那么好，我们可以用$\epsilon-greedy$的方法，前期$\epsilon$可以大一些随机性强一些，以后$\epsilon$调到低一些即可。

![image-20210511222437412](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210511222437412.png)



**Q-Learning vs. Sarsa:**

![image-20210511223537205](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210511223537205.png)

从backup这个图上来看：

![image-20210511223856072](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210511223856072.png)

Sarsa中的A' 是通过和A一样，从策略当前的策略抽取到的。而Q-Learning中，他是通过观察Q-Table，找到一个可以使得$Q(S_{t+1},a)$最大的$a$。



**cliff walk问题中，看一下Sarsa和Q-learning的区别：**

![image-20210511224316635](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210511224316635.png)

Q-learning的策略更激进，Sarsa的策略更稳健。



**总计：**

![image-20210511224418317](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210511224418317.png)