


[toc]





# 
- 输入和输出和权重都是NHWC哦！
- 结果矩阵是：(N * OH * OW) * (OC)是行存储矩阵
- 我们的每个threadBlock计算一个`blockM * blockN`这么大的输出哦！
  - 并且threadBlock中的每个线程只计算`blockM * blockN`的一个数字

&emsp;


- A * B
- A：(N * OH * OW) * (IC * FH * FW)
- B：(IC * FH * FW) * (OC):是行矩阵


&emsp;

- 理论上来讲，A矩阵从全局内存中搬运到smem中的时候，需要重新排布的，排布成矩阵的形式
  - 但是B矩阵似乎直接搬到smem就好了


