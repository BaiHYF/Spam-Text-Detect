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
