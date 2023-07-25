1. 提交任务时候需要申请足够的 gpu `--gres=gpu:4`
2. pl.Trainer 初始化需要设置参数和并行计算策略： `gpus=4, strategy="ddp_find_unused_parameters_false"`
3. 错误：
```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can ena
ble unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
making sure all `forward` function outputs participate in calculating loss.
```
本质原因是因为模型中有部分参数未参与梯度反传，需要保证模型所有参数都参与了计算。


找到没有回传迭代部分：https://blog.csdn.net/shaojie_45/article/details/123029735?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-123029735-blog-128558684.235%5Ev38%5Epc_relevant_anti_t3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-123029735-blog-128558684.235%5Ev38%5Epc_relevant_anti_t3&utm_relevant_index=3
辅助 loss：https://blog.csdn.net/qq_45717425/article/details/128558684


```python

# 去判断哪些参数没有梯度
for name, param in model.named_parameters():
    if param.grad is None:
        print(name)
```

可以进行 debug， 或者如果 multi task train 无法直接定义 module -> loss，可以在 `training_step()` 方法下加入（普通版本 `loss.backward()` 前）：
```python
for name, param in self.model.named_parameters():
    # if param.grad == None:
    loss += (param * 0).sum().to(param.device)
```




4. iterable dataset + ddp: 
The `__iter__` function is called by the data loader, for each worker when the data loader is first looped over. Have you tried returning `iter(list(mapped_itr)[uid::self.batch_size])`