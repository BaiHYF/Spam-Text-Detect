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