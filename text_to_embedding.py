from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

#信息熵
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