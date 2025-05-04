# 多尺度自适应阈值 (Multi-scale Adaptive Threshold)

## 功能说明

本修改为SAST (Scene Adaptive Sparse Transformer) 模型实现了多尺度自适应阈值功能，使不同stage的剪枝阈值可以单独调整和学习。

## 技术实现

1. 在`AdaptiveThresholdLearner`类中添加了对多stage的支持：
   - 为每个stage创建独立的基础阈值参数 (`base_bounces`)
   - 为每个stage创建独立的温度参数 (`temperatures`)
   - 为每个stage添加缩放因子 (`stage_scales`)，允许不同stage有不同的阈值权重

2. 在`SAST_block`类中添加了`stage_id`参数，用于标识当前block所属的stage

3. 在`SASTAttentionPairCl`和`RNNDetectorStage`类中传递stage_id参数，确保阈值学习器能够使用正确的stage参数

## 配置方法

在配置文件中添加以下参数：

```yaml
attention:
  # 其他参数...
  num_stages: 4  # 总stage数
  threshold_lr_scale: 0.01  # 阈值学习器学习率缩放因子
```

## 优势

1. **不同特征尺度适应性**：不同stage的特征尺度和语义层次不同，可以使用不同的阈值进行稀疏化

2. **细粒度控制**：可以针对不同stage的计算复杂度进行精细调整

3. **灵活性**：通过配置文件可以轻松调整不同stage的初始阈值和学习率

## 使用方法

模型会自动检测当前处理的stage ID，并使用对应的阈值参数。不需要额外的代码修改，只需在配置文件中设置`num_stages`参数即可。

## 注意事项

1. 如果要调整不同stage的初始阈值，可以在训练前手动修改`self.base_bounces`、`self.temperatures`和`self.stage_scales`参数

2. 在较深的stage中，由于特征已经高度抽象，可能需要更高的阈值来保持信息密度 