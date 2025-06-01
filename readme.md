# 中文垃圾文本分类系统

## 项目概述

本项目实现了一个高效、鲁棒的中文垃圾文本分类系统，通过创新的特征工程和优化的计算框架，在多个数据集上实现了99%以上的准确率和召回率。系统可广泛应用于电子邮件垃圾过滤、社交媒体内容审核、广告和欺诈信息识别等场景。

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch
- scikit-learn
- joblib
- pypinyin
- .........

### 安装

```
pip install -r requirements.txt
```

### 训练/使用模型

```bash
python main.py	
python main_light.py	#使用轻量化模型
```

### 使用Web界面

```
python app.py
```

## 系统特色

- **创新的特征表示方法**：融合汉字结构特征(字形、四角编码)与语义特征，引入文本自相似性和信息熵等统计特征
- **高性能计算架构**：实现缓存机制加速特征计算，支持多核并行处理，提供CPU/GPU双版本分类器
- **优异的分类性能**：在16k数据集上达到99.41%算术平均f1-score，在800k大规模数据集上保持99.46%算数平均f1-score
- **优秀的鲁棒性**：针对OOV字符和样本失衡问题进行了专门优化
- **完善的工程实现**：提供可视化Web界面，支持端侧轻量化部署

## 系统架构

```
前端(可视化层) + 后端(完整的垃圾文本分类算法)
```

核心算法分为四个模块，形成端到端的分类流水线:

1. **数据预处理**：加载原始文本，进行清洗与标准化
2. **字向量与编码**：基于汉字结构编码生成相似性矩阵，PCA降维生成词向量
3. **文本向量与特征工程**：融合语义向量与统计特征
4. **分类器**：双层全连接神经网络，支持CPU/GPU双版本

## 性能指标

| 数据集     | 准确率 | 召回率 | F1-score |
| ---------- | ------ | ------ | -------- |
| 16k数据集  | 99.41% | 99.41% | 99.41%   |
| 800k数据集 | 99.46% | 99.46% | 99.46%   |
| 混合数据集 | 97.91% | 97.91% | 97.90%   |

### **项目根目录结构**

```
./
├── **核心代码文件**
│   ├── app.py                     # Web应用入口（Flask/Django等）
│   ├── main.py                    # 主程序入口（完整版）
│   ├── main_light.py              # 轻量化主程序入口
│   ├── character_coder.py         # 汉字编码处理
│   ├── data_processing.py         # 数据预处理模块
│   ├── similarity_matrix.py       # 相似度矩阵计算
│   └── text_to_embedding.py       # 文本向量化模块

├── **数据文件**
│   ├── Data/                      # 原始数据目录
│   │   ├── dataset.txt            # 原始数据集
│   │   ├── dataset_new.txt        # 新扩展数据集
│   │   ├── merged_dataset.txt     # 合并后的数据集
│   │   ├── chinese_characters_code.txt    # 汉字编码表（完整版）
│   │   ├── chinese_characters_code_light.txt  # 轻量化编码表
│   │   ├── hanzijiegou_2w.txt     # 汉字结构数据
│   │   ├── hit_stopwords.txt      # 停用词表
│   │   ├── similarity_matrix.pkl  # 完整相似度矩阵
│   │   └── similarity_matrix_light.pkl  # 轻量化相似度矩阵

├── **模型文件**
│   ├── mlp_model.pth              # PyTorch模型权重
│   ├── model/                     # 模型相关文件
│   │   ├── model.pkl              # 序列化模型（如sklearn）
│   │   ├── char_embeddings.npy    # 字符嵌入向量
│   │   ├── char2idx.pkl           # 字符到索引映射
│   │   ├── label_encoder.pkl      # 标签编码器
│   │   ├── pca.pkl                # PCA降维模型
│   │   └── scaler.pkl             # 特征缩放器

├── **子模块**
│   ├── four_corner_method/        # 四角编码子模块
│   │   ├── __init__.py            # 模块初始化
│   │   ├── data/                  # 四角编码数据
│   │   └── __pycache__/           # 编译缓存

├── **Web相关**
│   ├── templates/                 # HTML模板
│   │   └── index.html             # 前端页面

├── **缓存与编译文件**
│   ├── __pycache__/               # Python编译缓存
│   │   ├── *.cpython-38.pyc       # Python 3.8缓存
│   │   └── *.cpython-312.pyc      # Python 3.12缓存

└── **文档**
    └── readme.md                  # 项目说明文档
```

1. **`Data/`**
   - 存放所有原始和预处理数据，包括：
     - 数据集文件（`dataset*.txt`）
     - 汉字编码表（`chinese_characters_code*.txt`）
     - 预计算的相似度矩阵（`similarity_matrix*.pkl`）
2. **`model/`**
   - 包含模型训练后的持久化文件：
     - `mlp_model.pth`：PyTorch神经网络模型权重
     - `model.pkl`：传统机器学习模型（如随机森林）
     - 特征工程相关文件（PCA、字符嵌入等）
3. **`four_corner_method/`**
   - 实现汉字四角编码方法的子模块，可能用于特征增强。
4. **`templates/`**
   - Web应用的前端模板（如Flask的HTML文件）。





