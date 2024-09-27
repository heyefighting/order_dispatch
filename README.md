# MOSC
该仓库开源了我本科阶段的一段meaningful科研经历的工作，它的前身是我发表的第一篇CCF论文[WASA, 2022]"Toward Multi-sided Fairness: A Fairness-Aware Order Dispatch System for Instant Delivery Service"，后来在此基础上完善，我又完成了本科毕业设计"基于多方公平的即时配送场景派单算法设计与实现"，提名江苏省优秀毕业设计，校优撒花❀
[论文链接](https://link.springer.com/chapter/10.1007/978-3-031-19214-2_25)

## 引用 Citation
```
@inproceedings{cao2022toward,
  title={Toward Multi-sided Fairness: A Fairness-Aware Order Dispatch System for Instant Delivery Service},
  author={Cao, Zouying and Jiang, Lin and Zhou, Xiaolei and Zhu, Shilin and Wang, Hai and Wang, Shuai},
  booktitle={International Conference on Wireless Algorithms, Systems, and Applications},
  pages={303--316},
  year={2022},
  organization={Springer}
}
```

## 论文总结 Summarization
### 系统利润效益比较
从即时配送平台角度来看，派单算法的派单质量不可或缺的一个评价指标便是其总体利润收益，因此本文将MOSC模型与其他基线模型进行比较。下图给出了不同派单算法下午高峰与晚高峰时段的系统利润收益变化。可以发现，所有方法由于强化学习技术的引入，与原始数据相比，都不同程度地提高了即时配送平台的总收益，其中MOSC模型以比GT增加了8.11%的优势胜过其他基准模型。
![不同派单算法下的午、晚高峰系统收益变化](https://github.com/user-attachments/assets/d104ddfb-d3ea-4de1-bcc1-a935c73f0d49)

### 多方公平性比较
本文的另一个关键目标是提高即时配送场景下多方利益相关者的公平性权益。下图1展示了从开始派单不断累积的骑手收入基尼系数变化，图2统计出基尼系数Gini与方差PF评价指标值，可见MOSC模型与PPO-Lagrangian方法不相上下，均有效提高了骑手利润公平性。
![骑手收入基尼系数变化](https://github.com/user-attachments/assets/5550201d-ac87-44a6-9c31-3381e7f9d543)

![不同派单算法下的骑手公平实验指标](https://github.com/user-attachments/assets/5f1de005-5c19-4820-9f53-11bd2723cca2)

但在商家收益差距比较方面，PPO-Lagrangian方法明显牺牲了MF指标（下图1），而MOSC依旧维持最佳性能。即使同样使用了多目标强化学习技术来平衡骑手与商家的利益，这些基线模型却不约而同地忽视了商家方的公平性诉求，唯有MOSC达到了比GT还低的商家收益差距，这可能归因于基于信赖域的优化方法稳定了模型学习过程。
同时，基于顾客订单超时率OV指标，本文比较了MOSC与所有基线模型下顾客订单超时时长的分布（下图2），MOSC在降低整体顾客等餐时间上最为显著，能将订单超时率控制到仅1.04%内。其次是引入拉格朗日乘子来考虑顾客公平性的两个方法，对OV指标的降低幅度相近。而A2CMF模型相比之下订单超时控制效果较差，验证了将顾客公平性问题作为强化学习框架的一部分进行建模求解的有效性。
 
 ![不同派单算法下的商家收益差距](https://github.com/user-attachments/assets/4baa2831-106b-4167-8d32-b233a9446c68)

![不同派单算法下的顾客订单超时分布](https://github.com/user-attachments/assets/fc0dfb78-7fd2-44f2-a5c2-9aa06852aef6)



