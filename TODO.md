## 概览
1. 统合 Matrix-City 的10个 Block 得到整个城市的Fine-Level GS
2. 基于 Fine-level GS 实现LoD功能，并展现在显存或速度上的性能优势

## 分工

### 刘洋
- [x] 在Block 9上拉通分块训练的管线
- [x] 在Block 3上拉通LoD的管线
- [ ] 在分块训练管线的基础上追加xyz范围限制，尽可能减少超参数改动以及额外的训练需求
- [ ] 在Block9上优化分块训练并行化，并测试新的分块训练配置的性能
- [ ] 优化Cell融合策略，消除交界处鬼影
- [ ] 统合Block1-Block10的所有数据得到整个城市的Fine Level GS


### 关赫
- [ ] 实现一版基于clamp的xyz范围限制，config参考 ```lod_mc_aerial_block3```，GaussianModel类别可以参考```GaussianModelVox```
- [ ] 训练并比较MatrixCity中定义的BlockA-BlockE上GS的性能，参看论文中```Table 2```，Block的划分方式记录于pose文件夹中，便于后续比较。主要指标包括3个metrics，GS数量以及渲染速度
- [ ] 设计并优化LoD的实现方案



