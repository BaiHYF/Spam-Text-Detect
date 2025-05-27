from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# 加载模型和资源
model = joblib.load('model/model.pkl')
pca = joblib.load('model/pca.pkl')
scaler = joblib.load('model/scaler.pkl')
le = joblib.load('model/label_encoder.pkl')
char2idx = joblib.load('model/char2idx.pkl')
char_embeddings = np.load('model/char_embeddings.npy')

# 标签映射（根据实际训练标签顺序调整）
LABEL_MAPPING = {0: '正常信息', 1: '垃圾信息'}

def text_to_embedding(text):
    """将文本转换为嵌入向量"""
    valid_chars = [c for c in text if c in char2idx]
    if not valid_chars:
        return None
    indices = [char2idx[c] for c in valid_chars]
    return np.mean(char_embeddings[indices], axis=0)

def calculate_contributions(text, embedding):
    """计算每个字符的贡献度"""
    if embedding is None:
        return []
    
    lr_coef = model.named_steps['logisticregression'].coef_[0]
    valid_chars = [c for c in text if c in char2idx]
    n_valid = len(valid_chars) or 1
    contributions = []
    total_score = 0
    amount = 0
    
    for pos, char in enumerate(text):

        if char not in char2idx:
            contributions.append({'position': pos, 'char': char, 'score': 0.0})
            continue
            
        # 计算单个字符贡献
        idx = char2idx[char]
        char_emb = char_embeddings[idx]
        
        # 正确计算标准化后的贡献
        scaled_emb = (char_emb/n_valid - scaler.mean_) / scaler.scale_
        score = np.dot(scaled_emb, lr_coef)
        total_score += score
        amount += 1
        contributions.append({'position': pos, 'char': char, 'score': float(score)})

    avg_score = total_score / amount
    
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
    if embedding is None:
        return jsonify({'label': '未知', 'probability': 0.0, 'contributions': []})
    
    # 预测处理
    X_scaled = scaler.transform([embedding])
    proba = model.predict_proba(X_scaled)[0]
    pred = np.argmax(proba)

    
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