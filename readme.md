# Spam Text Dectection

大数据课程作业

数据集来源：http://www.minerlab.cn/#/shared_resource/list

库安装：
- numpy
- sklearn
- pypinyin
- tqdm

运行程序：
```python3 main.py```

预期输出结果：
```
Dataset size:  10000
Tags:  {'1', '0'}
文本向量化: 100%|█████████████████████████████████████████| 10000/10000 [00:00<00:00, 44083.65it/s]

分类报告:
              precision    recall  f1-score   support

           0       0.82      0.92      0.87       624
           1       0.96      0.91      0.93      1376

    accuracy                           0.91      2000
   macro avg       0.89      0.92      0.90      2000
weighted avg       0.92      0.91      0.91      2000


混淆矩阵:
[[ 577   47]
 [ 129 1247]]
```

你们有什么修改直接push就行了，一个小作业不用搞那么麻烦。

##3.25模型优化
- 找到了更大的数据集
- 增加样本平衡机制，可以提高正确率和召回率，设置0.5~1.0之间
- 后期提升主要靠分类模型参数量的增加，例如全连接层从128-》256，能力迅速增强。
- 为解决分类器速度问题，使用GPU版本加速
- 调整PCA维度，143可以表征95%的信息



### 数据集
https://github.com/hrwhisper/SpamMessage/blob/master/data/%E5%B8%A6%E6%A0%87%E7%AD%BE%E7%9F%AD%E4%BF%A1.txt

### 结果
#### data_new 160k
完成训练

分类报告:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     28777
           1       0.91      0.90      0.90      3223

    accuracy                           0.98     32000
   macro avg       0.95      0.94      0.95     32000
weighted avg       0.98      0.98      0.98     32000


混淆矩阵:
[[28490   287]
 [  325  2898]]



#### data 16k
分类报告:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1000
           1       0.99      0.99      0.99      2200

    accuracy                           0.99      3200
   macro avg       0.99      0.99      0.99      3200
weighted avg       0.99      0.99      0.99      3200


混淆矩阵:
[[ 980   20]
 [  15 2185]]



## 3.26模型优化
- 类别权重+oversampleing双重，降低少类别分类错误
- persistent_workers=True提高数据传输速度

| 方法                       | 优点      | 适用情况       |
| ------------------------ | ------- | ---------- |
| mean\_pooling            | 快速、稳定   | 适合大多数任务    |
| weighted\_pooling        | 考虑词重要性  | 有 TF-IDF 时 |
| self\_attention\_pooling | 编码上下文关系 | 有足够长度文本    |
| max\_pooling             | 聚焦高强度特征 | 特征稀疏时效果好   |


## 模型解析
### 词向量生成模型
构建了一个汉字级词向量系统，词向量来自字符相似度矩阵的 PCA 分解。

### 句子向量生成模型
用字符向量经过 Self-Attention 聚合为句子向量，模拟 Transformer 中的上下文感知。

### 分类器



## 优化在新的数据集上的结果


# 全程流程优化
原始文本数据（Data/dataset.txt）
        ↓
数据划分（divide_dataset）
        ↓
[可选] 文本清洗（clean_text）
        ↓
字符统计/编码（char2idx, char_codes）
        ↓
字符相似度矩阵（sim_mat）
        ↓
PCA 降维 → 每个字符得到一个向量（char_embeddings）
        ↓
文本 → 句子向量（平均 or Self-Attention 聚合）
        ↓
句向量矩阵 X + 标签 y

## 特征工程
- PAC 语义
- len(special) 特殊字符长度
- len%(special) 特殊字符百分比




# to do list
- 有些汉字在编码里面不存在，直接跳过