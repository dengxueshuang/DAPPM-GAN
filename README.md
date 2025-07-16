# 无线电频谱补全：DAPPM + GAN

## 项目概述

本仓库实现了一种结合 **DAPPM** 模块与 **GAN**（生成对抗网络）的无线电频谱补全框架，旨在应对稀疏采样和非均匀衰减场景下的复杂无线环境。

**核心特点**

* **DAPPM 模块**：通过三条并行空洞卷积（膨胀率分别为 3、5、7）提取多尺度特征，并结合通道与位置注意力机制。
* **GAN 框架**：使用 ResNet 生成器和 CNN 判别器，生成高保真度的频谱重建结果。
* **自定义损失**：包括样式损失、感知损失、TV 损失、梯度一致性损失和功率谱损失。

## 代码结构

```text
DAPPM-GAN/
├── data/                  # 数据集加载及示例脚本
├── lib/
│   ├── loaders.py         # 数据加载与稀疏采样逻辑
│   ├── modules.py         # 网络模块定义：DAMF, RadioWNet
│   ├── EncoderModels.py   # GAN 生成器与判别器实现
│   └── loss.py            # 样式、感知、MS-SSIM 等损失函数及工具
├── loss/
│   ├── common.py          # 通用损失工具
│   └── loss.py            # 主损失实现
├── scripts/
│   └── train.py           # GAN 训练主循环脚本
├── outputs/               # 训练好的模型与日志（运行时生成）
└── README.md              # 项目说明（本文件）
```

## 环境安装

1. 克隆仓库：

   ```bash
   git clone https://github.com/yourusername/DAPPM-GAN.git
   cd DAPPM-GAN
   ```
2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

   * Python ≥3.7
   * PyTorch ≥1.8
   * torchvision
   * scikit-image、scipy、tqdm、pandas、matplotlib

## 使用方法

### 数据准备

将无线电频谱数据整理为 NumPy 数组或图像格式。编辑 `scripts/train.py` 中对 `loaders.RadioUNet_s` 的调用，设置：

* `fix_samples`：固定采样点数
* `num_samples_low`, `num_samples_high`：稀疏采样范围

### 训练

```bash
python scripts/train.py \
  --setup 1 \
  --batch_size 4 \
  --epochs 100 \
  --output_dir outputs/run1
```

* `--setup` 参数：

  1. `uniform`：固定 1% 采样
  2. `twoside`：一侧 1%，另一侧 10%
  3. `nonuniform`：1%–10% 非均匀采样
* 训练过程中，模型与日志保存在 `outputs/run1/` 目录。

### 评估

```bash
python evaluate.py \
  --model_path outputs/run1/Trained_ModelMSE_G.pt \
  --test_data path/to/test
```

根据需求实现 MSE、PSNR、谱距离等评估指标。

## 模型组件

### DAPPM 模块

1. **通道压缩**：1×1 卷积，将通道数从 `C` 压缩到 `C/r`，默认 `r=4`。
2. **多尺度空洞卷积**：三条深度可分离 3×3 卷积，膨胀率 `dilation=[3,5,7]`，再用 1×1 卷积恢复通道。
3. **通道注意力 (CAM)**：全局平均/最大池化 → 共享 MLP（1×1→ReLU→1×1）→ Sigmoid → 逐通道加权。
4. **位置注意力 (PAM)**：通道维度平均/最大池化 → 拼接 → 5×5 卷积 → Sigmoid → 按空间位置加权。
5. **特征融合与残差**：将 DAMF、CAM、PAM 输出相加后，经过可分离卷积 + Dropout + 1×1 卷积，然后与原始输入 `X` 做残差连接。

### GAN 框架

* **生成器 (Generator)**：`EncoderModels.py` 中的 `ResnetGenerator`，输入：2 通道（稀疏光谱 + 位置编码），输出：1 通道补全频谱。
* **判别器 (Discriminator)**：同文件中的 `Discriminator`，用于区分真/假频谱。
* **损失函数**：

  * 对抗损失 (`BCEWithLogitsLoss`)
  * 重建损失 (`L1Loss` 或 `MSELoss`)
  * 样式 & 感知损失
  * 总变差 (TV) 与梯度一致性
  * 功率谱差异 (FFT)

## 超参数配置

可在 `scripts/train.py` 或通过命令行修改：学习率、批大小、采样参数、损失权重等。

## 联系方式

如有问题，请提交 issue 或联系作者。
