# 说明

这是一个简单的使用 [PyTorch DistributedDataParallel](https://pytorch.org/tutorials/distributed/home.html#learn-ddp)
训练并评估模型的模板，包含以下内容:

- DDP 的简单使用样例 (见 train.py 与 eval.py)
- 将配置项放入 yaml 文件中
- 较完善的 logger, 确保仅在主进程上进行日志输出

任务为使用一个自定义的简单模型在 MNIST 数据集上进行分类

# 安装环境

## PyTorch 环境

使用 `nvidia-smi` 获取当前设备的 CUDA 版本

在 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取当前 CUDA 版本对应的 PyTorch 安装指令

例如 CUDA 版本为 12.6 时, 使用下述指令

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 其他依赖

运行

```shell
pip install -r requirements.txt
```

# 运行脚本

## 训练

注意 nproc_per_node 参数应当与配置文件中 CUDA_VISIBLE_DEVICES 设置的 GPU 数量一致

```shell
torchrun \
  --nproc_per_node 1 \
  --master_port 29501 \
  train.py \
  --config-path ./config/default.yaml 
```

## 测试
```shell
torchrun \
  --nproc_per_node 1 \
  --master_port 29501 \
  eval.py \
  --config-path ./config/default.yaml
```