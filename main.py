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
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


if __name__ == "__main__":
   # 1. 加载标签和文本（数据集划分）
   #dataset 16k  dataset_new 800k
    tags, texts = divide_dataset("Data/dataset_new.txt", lines=160000)
    
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
    
    # 分析 PCA 的累计解释方差
    # pca_full = PCA().fit(sim_mat)
    # explained_var_ratio = np.cumsum(pca_full.explained_variance_ratio_)
    # threshold = 0.95
    # recommended_dims = np.argmax(explained_var_ratio >= threshold) + 1
    # print(f"建议使用的 PCA 降维维度: {recommended_dims}（可保留 {threshold*100:.0f}% 的信息）")


    
    # 4. 对相似度矩阵进行PCA降维（得到每个汉字的向量表示）
    pca = PCA(n_components=150)
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

    # 使用 GPU（如可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

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
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
