"""
主程序模块

- Author: BaiHYF <baiheyufei@gmail.com>
- Date:   Mon May 19 2025

执行完整处理流程：
1. 数据加载与预处理
2. 特征编码生成
3. 相似度矩阵计算
4. 降维与分类建模
5. 模型评估


- Author: BIGHH <1448545037@qq.com>
- Date:   Mon May 26 2025
- 新增内容
- 找到了更大的数据集
- 增加样本平衡机制，可以提高正确率和召回率，设置0.5~1.0之间
- 后期提升主要靠分类模型参数量的增加，例如全连接层从128-》256，能力迅速增强。
- 为解决分类器速度问题，使用GPU版本加速
- 调整PCA维度，143可以表征95%的信息

"""

from data_processing import *
from character_coder import *
from similarity_matrix import *
from text_to_embedding import *


from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from joblib import Parallel, delayed
import os
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":

    # 使用 GPU（如可用）
    device = torch.device("cpu") 
    print(f"使用设备: {device}")

    # 1. 加载标签和文本（数据集划分）
    #dataset 16k  dataset_new 800k
    # tags, texts = divide_dataset("Data/dataset_new.txt", lines=500000)
    #dataset_path = "Data/merged_dataset.txt"
    dataset_path = "Data/dataset.txt"
    #dataset_path = "Data/dataset_new.txt"
    dataset_lines = 800000
    tags, texts = divide_dataset(dataset_path, lines=dataset_lines)
    
    print(f"Using  dataset: {dataset_path}, {dataset_lines} lines")

    
    # 2. 加载或统计汉字及其编码 [音] + [结构 + 四角编码 + 笔画数]
    try:  
        chinese_chars, chinese_chars_count, char_codes = load_chinese_characters("Data/chinese_characters_code_light.txt")
    except FileNotFoundError:
        chinese_chars, chinese_chars_count, char_codes = count_chinese_characters(texts, "Data/chinese_characters_code_light.txt")  #生成汉字编码
    
    # 3. 加载或计算汉字相似度矩阵 计算字符之间的相似性矩阵来表示网络关系。
    try:
        sim_mat = load_sim_mat("Data/similarity_matrix_light.pkl")
    except FileNotFoundError:
        sim_mat = compute_sim_mat(chinese_chars, char_codes,"Data/similarity_matrix_light.pkl")
    
    # 4. 对相似度矩阵进行PCA降维（得到每个汉字的向量表示）
    pca = PCA(n_components=128) #128，192,256
    char_embeddings = pca.fit_transform(sim_mat)
    
    # 建立汉字到向量索引的映射
    char2idx = {char: i for i, char in enumerate(chinese_chars)}
    
    # 5. 生成文本向量
    X = np.array(Parallel(n_jobs=32)(delayed(text_to_embedding)(text, pca, char_embeddings, char2idx) for text in tqdm(texts, desc="并行向量化")))
    print(X.shape)
    
    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(tags)
    
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 进行训练集的过采样（样本平衡）
    unique, counts = np.unique(y_train, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print("原始训练集中每个标签的样本数量:")
    for label, count in label_counts.items():
        print(f"标签 {label}: {count} 条样本")
    oversampler = RandomOverSampler(sampling_strategy=0.7, random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print("样本平衡后训练集中每个标签的样本数量:")
    for label, count in label_counts.items():
        print(f"标签 {label}: {count} 条样本")
    print("完成样本平衡")

    if device == torch.device("cpu"):
        print("使用CPU训练")
        #######CPU版本
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),  # 2层：128 -> 64
                activation="relu",
                solver="adam",
                batch_size=64,
                max_iter=20,
                early_stopping=True,
                random_state=42,
                verbose=True
            )
        )
        model.fit(X_train, y_train)
        print("完成训练")
        scaler = model.named_steps['standardscaler']
        y_pred = model.predict(X_test)
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=le.classes_,digits=4))
        print("\n混淆矩阵:")
        print(confusion_matrix(y_test, y_pred))
        #######CPU版本#####END

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(pca, 'model/pca.pkl')
    joblib.dump(le, 'model/label_encoder.pkl')
    joblib.dump(char2idx, 'model/char2idx.pkl')
    np.save('model/char_embeddings.npy', char_embeddings)
    joblib.dump(scaler, 'model/scaler.pkl')
        
