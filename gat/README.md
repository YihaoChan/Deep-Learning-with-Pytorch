# 图注意力网络

## 总结

1. `model.py`和`layer.py`是自己的复现；

2. 看论文时有几个细节，比如mask attention，只是一句话带过，这部分要结合开源代码才能真正明白，自己实现的话，确实没有想到；

3. 论文中的公式和真正代码实现的时候存在gap，比如计算注意力系数的时候：

   `a^T * [Whi || Whj]`

   如果按照这个公式编程，(在我看来)应该是做不出来的...必须调换顺序才能满足维度条件，然后还要拆成两部分相加，利用广播机制才能实现`[N, N]`的注意力系数矩阵，和`[N, N]`的邻接矩阵做mask。

## 参考资料

1. https://arxiv.org/pdf/1710.10903.pdf
2. https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/graph_attention_layer.py
3. https://zhuanlan.zhihu.com/p/81350196
4. https://blog.csdn.net/weixin_43476533/article/details/107229242
5. https://zhuanlan.zhihu.com/p/128072201
6. https://zhuanlan.zhihu.com/p/112938037