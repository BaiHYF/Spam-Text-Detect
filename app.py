from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
import math
from collections import Counter

app = Flask(__name__)

# 加载模型和资源
model = joblib.load('model/model.pkl')
pca = joblib.load('model/pca.pkl')
scaler = joblib.load('model/scaler.pkl')
le = joblib.load('model/label_encoder.pkl')
char2idx = joblib.load('model/char2idx.pkl')
char_embeddings = np.load('model/char_embeddings.npy')

LABEL_MAPPING = {0: '正常信息', 1: '垃圾信息'}

import math
from collections import Counter
def text_entropy(text):
    if not text or not isinstance(text, str):
        return 0.0 
    
    if len(text) == 1:
        return 0.0 
    
    freq = Counter(text)
    total = len(text)
    entropy = 0.0
    
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def self_similarity(text):
    if not isinstance(text, str) or len(text) < 4:
        return 0.0

    chunk_size = min(15, max(4, len(text)//5))
    overlap = max(1, chunk_size // 2)
    chunks = [
        text[i:i+chunk_size] 
        for i in range(0, len(text)-chunk_size+1, overlap)
    ]
    if len(chunks) < 2:
        return 0.0
    try:
        tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1,2)).fit_transform(chunks)
        sim_matrix = cosine_similarity(tfidf)   
        np.fill_diagonal(sim_matrix, 0)
        return sim_matrix.sum() / (len(chunks)*(len(chunks)-1))
    except:
        return 0.0
    
stopwords_file = 'Data/hit_stopwords.txt'
with open(stopwords_file, 'r', encoding='utf-8') as file:
    stopwords = {line.strip() for line in file}

def clean_input_text(text):
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text).strip()
    processed = ''.join([
        char for char in cleaned 
        if char not in stopwords and re.search("[\u4e00-\u9fa5]", char)
    ])
    return processed


def text_to_embedding(text):

    vec_semantic = np.zeros(pca.n_components_)

    valid_chars = [char for char in text if char in char2idx]
    if valid_chars:
        indices = [char2idx[char] for char in valid_chars]
        vec_semantic = np.mean(char_embeddings[indices], axis=0)

    special_chars = re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9]', text)
    special_ratio = len(special_chars) / max(1, len(text)) 

    vec_combined = np.concatenate([
        vec_semantic,             
        [special_ratio],           
        [len(special_chars)],      
        [self_similarity(text)],    
        [text_entropy(text)],       
    ])
    return vec_combined


def calculate_contributions(text, embedding):
    """计算每个字符的贡献度"""
    if embedding is None:
        return [], 0.0
    
    # 获取MLP模型结构和权重
    mlp = model.named_steps['mlpclassifier']
    
    # 输入层到第一个隐藏层的权重矩阵
    input_weights = mlp.coefs_[0]  
    
    # 计算特征重要性
    feature_importance = np.linalg.norm(input_weights, axis=1)
    
    # 标准化后的特征值
    scaled_emb = scaler.transform([embedding])[0]
    
    contributions = []
    total_contribution = 0
    valid_count = 0
    
    # 处理语义特征部分（前256维）
    valid_chars = [c for c in text if c in char2idx]
    n_valid = len(valid_chars) if valid_chars else 1
    
    for pos, char in enumerate(text):
        char_contribution = 0.0
        
        if char in char2idx:
            # 语义特征贡献度
            semantic_contribution = np.sum(
                feature_importance[:256] * scaled_emb[:256]
            ) / n_valid
            
            # 结构特征平均分配
            structure_contribution = np.sum(
                feature_importance[256:] * scaled_emb[256:]
            ) / max(len(text), 1)
            
            char_contribution = semantic_contribution + structure_contribution
            valid_count += 1
        else:
            # 非中文字符只分配结构特征
            structure_contribution = np.sum(
                feature_importance[256:] * scaled_emb[256:]
            ) / max(len(text), 1)
            char_contribution = structure_contribution
        
        contributions.append({
            'position': pos, 
            'char': char, 
            'score': float(char_contribution)
        })
        total_contribution += char_contribution
    
    avg_score = total_contribution / len(text)
    
    return contributions, avg_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '').strip()
    if not text:
        return jsonify({'error': '输入不能为空'})
    
    # 生成嵌入向量
    embedding = text_to_embedding(text)
    
    # 预测处理
    X_scaled = scaler.transform([embedding])
    proba = model.predict_proba(X_scaled)[0]
    pred = np.argmax(proba)
    print(proba)
    # 获取贡献度
    contributions, avg_score = calculate_contributions(text, embedding)
    
    return jsonify({
        'label': LABEL_MAPPING.get(pred, '未知'),
        'probability': float(proba[pred]),
        'contributions': contributions,
        'avg': avg_score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)