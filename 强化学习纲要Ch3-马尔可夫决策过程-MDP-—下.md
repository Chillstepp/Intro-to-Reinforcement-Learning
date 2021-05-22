---
title: 强化学习纲要Ch3-马尔可夫决策过程(MDP)—下
date: 2021-05-20 21:58:01
index_img: /img/rl_zbl.png
tags: 强化学习
---

# 马尔可夫决策过程(MDP)—下

**马尔可夫决策过程有两个核心问题：分别是预策(prediction)和控制(control)**

- 预测：

![image-20210503214530939](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210503214530939.png)

预策问题就是给定马尔可夫决策过程和策略$\pi$， 或者给出马尔科夫奖励过程。然后去做预策每个状态的价值函数$v^{\pi}$

- 控制：

![image-20210503214841239](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210503214841239.png)

控制是指给出一个马尔可夫过程，需要最优化得到价值函数和策略。



这两个问题都可以通过动态规划来解决。



首先是预测问题：

![image-20210503215450158](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210503215450158.png)

通过这样的状态转移方程去做动态规划即可。

这里给出一个代码框架：
$$
v_{t+1}(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in \mathcal{S}} P\left(s^{\prime} \mid s, a\right) v_{t}\left(s^{\prime}\right)\right)
$$

```python
#在已知策略\pi 下迭代
while True:
    old_table = self.table.copy()

    for state in range(self.obs_dim):
        #由于已知策略，可以直接根据policy，输入当前state找出act
        act = self.policy(state)
        transition_list = self._get_transitions(state, act)

        state_value = 0
        for transition in transition_list:
            prob = transition['prob']
            reward = transition['reward']
            next_state = transition['next_state']
            done = transition['done']

            # [TODO] what is the right state value?
            # hint: you should use reward, self.gamma, old_table, prob,
            # and next_state to compute the state value
            state_value += prob*(reward+self.gamma*1*old_table[next_state])

            # update the state value
            self.table[state] = state_value

        # [TODO] Compare the old_table and current table to
        #  decide whether to break the value update process.
        # hint: you should use self.eps, old_table and self.table
        should_break = None
        #如果迭代两次差距太小就说明收敛
        if (np.fabs(self.table - old_table) < np.array([self.eps]*self.obs_dim)).all():
            should_break = True
```

![image-20210503222023509](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210503222023509.png)



我们举一个例子来迭代一下$v$:

开始时:

![image-20210503222552287](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210503222552287.png)

一定次数的传播后$v$就会逐渐稳定：

![image-20210503222638284](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210503222638284.png)



上述都是再讲怎么迭代计算出$v^{\pi}(s)$。

我们还有一个问题是control，也就是如何算出最优化的策略/状态价值函数：

![image-20210503224251512](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210503224251512.png)

最优的状态价值函数中的有关变量是策略$\pi$, 也就是 在各种各样的策略下，可以使得每个状态价值最大的策略就是我们需要选择的策略，在这样的策略下，我们就可以获得最优的状态价值函数。

![image-20210503225215570](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210503225215570.png)

同理，如果找最优策略，就如上述所说的，找可以使得状态价值函数最大的策略就被成为最优的策略。

当我们知道最优化的值时，就可以说MDP问题被解决了。

**怎么寻找最佳的策略：**

![image-20210504091803979](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210504091803979.png)

最简单的一种方式就是选择可以使得$a =  argmax _{a \in A}q*(s,a)$的action，这样的action，并让这种动作的在状态出现的概率$\pi^*(a \mid s) = 1$即可，此时由公式：
$$
v^{\pi}(s)=\sum_{a \in A} \pi(a \mid s) q^{\pi}(s, a)
$$
不难得知$v^{\pi}$在此时取得了最大值，那么根据上最优化策略的定义，此时的$\pi$即为最优策略。

一个简单的想法，就是枚举所有的状态的action，算一算他们的state-value function，然而这种方法仅仅枚举复杂度就达到了$O(s^a)$，其中s为状态数，a为在该状态下可做的行为数量。

还有一种想法是**策略迭代(Policy Iteration)：**

![image-20210504094510039](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210504094510039.png)

这种策略分为两部分，一个是计算在策略$\pi$下的value function，另一个是提升策略(即贪心的选择可以使得v函数更大的$\pi$)。

提升策略中有两步：

![image-20210504094854696](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210504094854696.png)

首先是计算状态行为价值函数(q函数)，也就是在状态s做出动作a所获得的reward。

然后计算新策略下计算使得q在状态s下可以被最大化的动作。这种迭代方式叫做策略迭代：它包括策略估计和策略提升



上面已经给出在策略$\pi$下如何迭代计算出价值函数，这里再给出**策略提升的框架代码：**

```python
    def update_policy(self):
        """You need to define a new policy function, given current
        value function. The best action for a given state is the one that
        has greatest expected return.

        To optimize computing efficiency, we introduce a policy table,
        which take state as index and return the action given a state.
        """
        policy_table = np.zeros([self.obs_dim, ], dtype=np.int)

        for state in range(self.obs_dim):
            state_action_values = [0] * self.action_dim
            
            # [TODO] assign the action with greatest "value"
            # to policy_table[state]
            # hint: what is the proper "value" here?
            #  you should use table, gamma, reward, prob,
            #  next_state and self._get_transitions() function
            #  as what we done at self.update_value_function()
            #  Bellman equation may help.
            best_action = None
            for act in range(self.action_dim):
                transition_list = self._get_transitions(state, act)
                action_value = 0
                for transition in transition_list:
                    prob = transition['prob']
                    reward = transition['reward']
                    next_state = transition['next_state']
                    done = transition['done']
                    action_value += prob*(reward+self.gamma*self.table[next_state])
                state_action_values[act] = action_value
            best_action = np.argmax(state_action_values)
            policy_table[state] = best_action

        self.policy = lambda obs: policy_table[obs]
```



那么就可以推出：Bellman Optimality Equation

![image-20210504101827815](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210504101827815.png)

因此就可以通过**Bellman Optimality Equation得出迭代算法：**

这种迭代叫做：**价值迭代Value iteration**![image-20210504102033826](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210504102033826.png)

代码框架：

```python
    old_policy_result = {
        obs: -1 for obs in range(trainer.obs_dim)
    }
    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()  # [TODO] please uncomment this line

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print("[INFO]\tIn {} iteration, current "
                  "mean episode reward is {}.".format(
                i, trainer.evaluate()
            ))

            # [TODO] compare the new policy with old policy to check should
            #  we stop.
            # [HINT] If new and old policy have same output given any
            #  observation, them we consider the algorithm is converged and
            #  should be stopped.
            should_stop = None
            should_stop = True
            new_policy_result = {
            obs: trainer.policy(obs) for obs in range(trainer.obs_dim)
            }
            for obs in range(trainer.obs_dim):
                if new_policy_result[obs] != old_policy_result[obs]:
                    should_stop = False
                    break
            old_policy_result = new_policy_result
            if should_stop:
                print("We found policy is not changed anymore at "
                      "iteration {}. Current mean episode reward "
                      "is {}. Stop training.".format(i, trainer.evaluate()))
                break
```

举一个最短路的例子，在不断的更新迭代，我们发现距离左上角深色方块越远，v函数在该点的状态越小。

![image-20210504102959224](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210504102959224.png)





**最后总结一下策略迭代和价值迭代的区别：**

![image-20210504104405231](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210504104405231.png)

![image-20210504104546069](https://gitee.com/Chillstep/ChillstepPictures/raw/master/master/image-20210504104546069.png)