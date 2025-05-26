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
from similarity_matrix import compute_sim_mat, load_sim_mat
from character_coder import ChineseCharacterCoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import time
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import sys

from joblib import Parallel, delayed

# 文本清洗
def clean_text(dataset):
    cleaned_text = []
    for text in tqdm(dataset, desc='Cleaning text'):
        clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        cleaned_text.append(clean.strip())
    return cleaned_text

# 停用词处理和文本分割
def tokenize_and_remove_stopwords(dataset):
    stopwords_file = '第四章/高阶示例/数据集/hit_stopwords.txt'
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = {line.strip() for line in file}
    tokenized_text = []
    for text in tqdm(dataset, desc='Tokenizing and removing stopwords'):
        cleaned_text = ''.join([char for char in text if char not in stopwords and re.search("[\u4e00-\u9fa5]", char)])
        tokenized_text.append(cleaned_text)
    return tokenized_text

# 定义文本转向量函数：对每个字符向量取平均
# def text_to_embedding(text):
#     vec = np.zeros(pca.n_components_)
#     valid_chars = [char for char in text if char in char2idx]   #仅支持汉字以及在字典里的字
#     if not valid_chars:
#         return vec
#     indices = [char2idx[char] for char in valid_chars]
#     return np.mean(char_embeddings[indices], axis=0)


#信息熵
import math
from collections import Counter
def text_entropy(text):
    if not text or not isinstance(text, str):
        return 0.0  # 处理空文本或非字符串输入
    
    if len(text) == 1:
        return 0.0  # 单字符文本熵为0
    
    freq = Counter(text)
    total = len(text)
    entropy = 0.0
    
    for count in freq.values():
        p = count / total
        if p > 0:  # 避免log2(0)
            entropy -= p * math.log2(p)
    
    return entropy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def self_similarity(text):
    """ 改进版自相似性计算 """
    # 边界检查
    if not isinstance(text, str) or len(text) < 4:
        return 0.0
    # 动态分块（4~15字，50%重叠）
    chunk_size = min(15, max(4, len(text)//5))
    overlap = max(1, chunk_size // 2)
    chunks = [
        text[i:i+chunk_size] 
        for i in range(0, len(text)-chunk_size+1, overlap)
    ]
    if len(chunks) < 2:
        return 0.0
    try:
        # 字符级TF-IDF（支持中文）
        tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1,2)).fit_transform(chunks)
        sim_matrix = cosine_similarity(tfidf)   
        # 排除自相似
        np.fill_diagonal(sim_matrix, 0)
        return sim_matrix.sum() / (len(chunks)*(len(chunks)-1))
    except:
        return 0.0  # 处理全相同字符等异常



def text_to_embedding(text):
    # 初始化PCA字向量的均值（语义部分）
    vec_semantic = np.zeros(pca.n_components_)
    # 1. 提取有效汉字并计算语义向量
    valid_chars = [char for char in text if char in char2idx]  # 仅处理字典中的汉字
    if valid_chars:
        indices = [char2idx[char] for char in valid_chars]
        vec_semantic = np.mean(char_embeddings[indices], axis=0)  # 字向量均值
    # 2. 计算特殊符号占比特征（结构部分）
    special_chars = re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9]', text)  # 排除空格
    special_ratio = len(special_chars) / max(1, len(text))  # 符号占比
    # 3. 融合语义向量和符号特征
    vec_combined = np.concatenate([
        vec_semantic,               # PCA字向量均值（语义）
        [special_ratio],           # 特殊符号占比（结构）
        [len(special_chars)],      # 绝对符号数量
        [self_similarity(text)],    # 自相似性
        [text_entropy(text)],       #信息熵
    ])
    return vec_combined


############# 词向量-》句子向量 version-2
def text_to_embedding_self_attention(text, char_embeddings, char2idx, d=256):
    """
    使用 Self-Attention 聚合字符向量，生成句子向量。
    
    参数：
        text: str，输入文本
        char_embeddings: np.ndarray，所有汉字向量，shape=(n_chars, d)
        char2idx: dict[str -> int]，字符到索引的映射
        d: int，向量维度
        
    返回：
        np.ndarray，句子向量，shape=(d,)
    """
    # 筛选出有效字符及其向量
    valid_chars = [char for char in text if char in char2idx]   ##仅支持汉字以及在字典里的字
    if not valid_chars:
        return np.zeros((d,))

    # 获取字符向量矩阵 V
    char_vecs = np.array([char_embeddings[char2idx[c]] for c in valid_chars])  # shape: (n, d)
    n = char_vecs.shape[0]

    # Step 1: 构建 Self-Attention 权重矩阵（缩放点积）
    scores = np.dot(char_vecs, char_vecs.T) / np.sqrt(d)  # shape: (n, n)

    # Step 2: softmax over rows → attention weights
    scores -= np.max(scores, axis=1, keepdims=True)  # 防止溢出
    attention_weights = np.exp(scores)
    attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)  # shape: (n, n)

    # Step 3: 每个字符向量加权上下文
    attended_vecs = np.dot(attention_weights, char_vecs)  # shape: (n, d)

    # Step 4: 所有位置平均池化成句子向量
    sentence_vector = np.mean(attended_vecs, axis=0)  # shape: (d,)
    return sentence_vector


# 根据字符相似性网络生成最终的字嵌入向量
# 参数说明：
#   - chinese_characters: 所有待处理的中文字符列表
#   - w2v_vectors: 预训练的字向量字典（Word2Vec格式）
#   - sim_mat: 字符相似度矩阵（N×N的对称矩阵）
#   - text: 原始文本数据（用于扩展未登录词）
#   - chinese_characters_count: 字符在语料中的出现频次统计
#   - threshold: 相似度阈值（默认0.6），用于判定字符是否相似
def generate_char_vectors(chinese_characters, w2v_vectors, sim_mat, text, chinese_characters_count, threshold=0.6):
    # 初始化存储最终字符向量的字典
    char_vectors = {}
    # 遍历所有中文字符（显示进度条）
    for i in tqdm(range(len(chinese_characters)), desc='Generating char vectors'):
        character = chinese_characters[i]  # 当前处理的字符
        similar_group = []  # 存储与当前字符相似的字符组
        # 遍历相似度矩阵的当前行，找出相似字符
        for j in range(len(sim_mat[i])):
            if sim_mat[i][j] >= threshold:  # 如果相似度超过阈值
                similar_group.append(chinese_characters[j])  # 加入相似组
        # 初始化加权平均向量（维度与w2v_vectors保持一致）
        sum_count = 0  # 相似组字符的总频次
        emb = np.zeros_like(w2v_vectors[list(w2v_vectors.keys())[0]])  # 全零初始化
        # 对相似组中的每个字符进行加权平均
        for c in similar_group:
            # 如果字符不在预训练向量中，动态更新词表
            if c not in w2v_vectors.keys():
                update(w2v_vectors, text, c)  # 调用未展示的update函数扩展词表
            # 按字符频次加权累加向量
            emb += chinese_characters_count[c] * w2v_vectors[c]
            sum_count += chinese_characters_count[c]
        # 计算加权平均值（防止除以0）
        emb /= sum_count if sum_count else 1
        # 将最终向量存入字典
        char_vectors[character] = emb
    return char_vectors

def special_char_ratio(text):
    # 使用正则表达式匹配所有非中文、非字母、非数字的字符
    non_chinese = re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9]', text)
    # 返回特殊字符数占总文本长度的比例（避免除以0）
    return len(non_chinese) / max(1, len(text))

if __name__ == "__main__":
    # 1. 加载标签和文本（数据集划分）
    #dataset 16k  dataset_new 800k
    tags, texts = divide_dataset("Data/dataset_new.txt", lines=500000)
    
    #预处理
    #cleaned_text = clean_text(texts)
    # print(texts[1])
    # print(cleaned_text[1])
    # tokenized_text = tokenize_and_remove_stopwords(cleaned_text)
    #texts=cleaned_text
    chinese_characters = []
    chinese_characters_count = {}
    chinese_characters_code = {}
    for line in tqdm(texts, desc="Counting characters", unit="line"):
        for char in line:
            if "\u4e00" <= char <= "\u9fff":  # 判断是否为汉字
                chinese_characters_count[char] = (
                    chinese_characters_count.get(char, 0) + 1
                )
    print(len(chinese_characters_count))
    
    # 2. 加载或统计汉字及其编码 [音] + [结构 + 四角编码 + 笔画数]
    try:  
        chinese_chars, _, char_codes = load_chinese_characters("Data/chinese_characters_code.txt")
    except FileNotFoundError:
        chinese_chars, _, char_codes = count_chinese_characters(texts, "Data/chinese_characters_code.txt")  #生成汉字编码
    #chinese_chars, _, char_codes=load_or_update_chinese_characters(texts, "Data/chinese_characters_code.txt")
    
    # 3. 加载或计算汉字相似度矩阵 计算字符之间的相似性矩阵来表示网络关系。
    try:
        sim_mat = load_sim_mat("Data/similarity_matrix.pkl")
    except FileNotFoundError:
        sim_mat = compute_sim_mat(chinese_chars, char_codes)
        
    # #分析 PCA 的累计解释方差
    # pca_full = PCA().fit(sim_mat)
    # explained_var_ratio = np.cumsum(pca_full.explained_variance_ratio_)
    # for threshold in [0.95,0.96,0.97,0.98]:
    #     recommended_dims = np.argmax(explained_var_ratio >= threshold) + 1
    #     print(f"建议使用的 PCA 降维维度: {recommended_dims}（可保留 {threshold*100:.0f}% 的信息）")
    
    # 4. 对相似度矩阵进行PCA降维（得到每个汉字的向量表示）
    pca = PCA(n_components=256) #128，192,256
    char_embeddings = pca.fit_transform(sim_mat)
    
    # 建立汉字到向量索引的映射
    char2idx = {char: i for i, char in enumerate(chinese_chars)}
    
    # # 字符向量聚合为词向量
    # words = jieba.lcut(text)  # ['您的', '账号', '已', '中奖', '请', '立即', '点击']
    # # 对每个词内的字符向量平均，得到词向量
    # word_vecs = []
    # for word in words:
    #     char_vecs = [char_embeddings[char2idx[c]] for c in word if c in char2idx]
    #     if char_vecs:
    #         word_vecs.append(np.mean(char_vecs, axis=0))
    
    '''5. 生成句子向量'''
    #X = np.array([text_to_embedding(text) for text in tqdm(texts, desc="文本向量化")])
    X = np.array(Parallel(n_jobs=32)(delayed(text_to_embedding)(text) for text in tqdm(texts, desc="并行向量化")))
    #X = np.array([text_to_embedding_self_attention(text, char_embeddings, char2idx, d=512)for text in tqdm(texts, desc="使用注意力生成句向量")])
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
    
    
    
    
    
    
    ########CPU版本
    # model = make_pipeline(
    #     StandardScaler(),
    #     MLPClassifier(
    #         hidden_layer_sizes=(128, 64),  # 2层：128 -> 64
    #         activation="relu",
    #         solver="adam",
    #         batch_size=64,
    #         max_iter=40,
    #         early_stopping=True,
    #         random_state=42,
    #         verbose=True
    #     )
    #     #LogisticRegression(max_iter=1000, class_weight='balanced')
    # )


    
    # model.fit(X_train, y_train)
    
    # print("完成训练")
    
    
    # y_pred = model.predict(X_test)
    
    # print("\n分类报告:")
    # print(classification_report(y_test, y_pred, target_names=le.classes_))
    # print("\n混淆矩阵:")
    # print(confusion_matrix(y_test, y_pred))
    
    
    
    
    
    
    
    
    
    ###########GPU训练
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    load_existing_model=False
    #torch.set_float32_matmul_precision('high')  #在 float32 矩阵乘法时，使用 TensorFloat32（TF32） 精度，能显著加快训练速度而不会明显影响模型精度。
    # 使用 GPU（如可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    
    # 归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 转为 Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # 计算类别权重，原理同oversampleing
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # print("类别权重为",class_weights)


    # 数据加载器
    train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=1024,
    shuffle=True,
    num_workers=16,   # 加速数据加载
    pin_memory=True, # 推荐用于 GPU 模型
    prefetch_factor=8,
    persistent_workers=True #避免反复启动线程，提高训练速度
    )

    # 定义 MLP 模型
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden1=256, hidden2=128, output_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(0.4),           # 添加 dropout
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden2, output_dim)
            )

        def forward(self, x):
            return self.net(x)


    model = MLP(input_dim=X.shape[1], output_dim=len(le.classes_)).to(device)
    
    # if load_existing_model:
    #     load_model(model)
    #model = torch.compile(model)    #编译，以进一步提高速度
    
    # 损失函数与优化器
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    for epoch in range(50):
        model.train()
        total_loss = 0
        y_train_pred_all = []
        y_train_true_all = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device,non_blocking=True), yb.to(device,non_blocking=True)   #异步传输
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        
        ####################添加更多调试信息##############################
        # 收集训练预测结果
        y_pred_batch = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        y_train_pred_all.extend(y_pred_batch)
        y_train_true_all.extend(yb.cpu().numpy())
        train_acc = accuracy_score(y_train_true_all, y_train_pred_all)
        model.eval()
        with torch.no_grad():
            logits = model(X_test_tensor.to(device))
            y_test_pred = torch.argmax(logits, dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    #关闭load线程
    train_loader._iterator._shutdown_workers()
    
    # 保存模型
    torch.save(model.state_dict(), "mlp_model.pth")
    print("模型已保存为 mlp_model.pth")
    print("完成训练")

    # 推理与评估
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_,digits=4))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
