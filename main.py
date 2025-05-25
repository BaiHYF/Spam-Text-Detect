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
"""

from data_processing import divide_dataset, count_chinese_characters, load_chinese_characters
from similarity_matrix import compute_sim_mat, load_sim_mat
from character_coder import ChineseCharacterCoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
   # 1. 加载标签和文本（数据集划分）
    tags, texts = divide_dataset("Data/dataset.txt", lines=16000)
    
    # 2. 加载或统计汉字及其编码
    try:  
        chinese_chars, _, char_codes = load_chinese_characters("Data/chinese_characters_code.txt")
    except FileNotFoundError:
        chinese_chars, _, char_codes = count_chinese_characters(texts, "Data/chinese_characters_code.txt")
    
    # 3. 加载或计算汉字相似度矩阵
    try:
        sim_mat = load_sim_mat("Data/similarity_matrix.pkl")
    except FileNotFoundError:
        sim_mat = compute_sim_mat(chinese_chars, char_codes)
    
    # 4. 对相似度矩阵进行PCA降维（得到每个汉字的向量表示）
    pca = PCA(n_components=100)
    char_embeddings = pca.fit_transform(sim_mat)
    
    # 建立汉字到向量索引的映射
    char2idx = {char: i for i, char in enumerate(chinese_chars)}
    
    # 定义文本转向量函数：对每个字符向量取平均
    def text_to_embedding(text):
        vec = np.zeros(pca.n_components_)
        valid_chars = [char for char in text if char in char2idx]
        if not valid_chars:
            return vec
        indices = [char2idx[char] for char in valid_chars]
        return np.mean(char_embeddings[indices], axis=0)
    
    # 5. 文本向量化
    X = np.array([text_to_embedding(text) for text in tqdm(texts, desc="文本向量化")])
    
    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(tags)
    
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),  # 2层：128 -> 64
            activation="relu",
            solver="adam",
            batch_size=64,
            max_iter=20,
            early_stopping=True,
            random_state=42
        )
        #LogisticRegression(max_iter=1000, class_weight='balanced')
    )


    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
