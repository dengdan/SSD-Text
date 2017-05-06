# 训练
## 参数设置
主要参数见`scripts/run.sh`
全部参数见`scripts/training_config.txt`
注意：
    1. 本实验设置中的aspect ratio没有改变
    2. training_config.txt不是配置文件， 每次执行训练时都会重新生成，修改无用。
## 启动训练
```
sh run.sh 0 train
```
其中， 0为gpu id.


