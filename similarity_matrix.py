"""
相似度矩阵模块

- Author: BaiHYF <baiheyufei@gmail.com>
- Date:   Mon May 19 2025

- Author: BIGHH <1448545037@qq.com>
- Date:   Mon May 28 2025

包含相似度矩阵的以下操作：
- 相似度的计算
- 矩阵计算与存储
- 矩阵加载
- 矩阵动态更新

Functions:
    compute_sim_mat(chinese_characters, chinese_characters_code) -> list[list[float]]
    load_sim_mat(filename) -> numpy.ndarray
    update_sim_mat(new_characters, chinese_characters_code, sim_mat) -> numpy.ndarray
    computeSoundCodeSimilarity(soundCode1, soundCode2) -> float
    computeShapeCodeSimilarity(shapeCode1, shapeCode2) -> float
    computeSSCSimilarity(ssc1, ssc2) -> float
    
模块依赖:
- numpy: 矩阵运算
- tqdm: 进度条显示
- pickle: 数据序列化

大部分程序参考了 《数据科学与工程实战》王昌栋，赖剑煌 第 4.3.1 节的代码实现     
"""


import numpy as np
from tqdm import tqdm
import pickle


# 声音和形状的权重
soundWeight=0.5
shapeWeight=0.5

# 计算字音编码相似性的函数
def computeSoundCodeSimilarity(soundCode1, soundCode2):
    # 特征大小（声音编码的长度）
    featureSize=len(soundCode1)
    # 特征权重
    weights=[0.4,0.4,0.1,0.1]
    multiplier=[]
    # 计算每个特征的相似性
    for i in range(featureSize):
        if soundCode1[i]==soundCode2[i]:
            multiplier.append(1)
        else:
            multiplier.append(0)
    soundSimilarity=0
    # 计算声音编码的相似性
    for i in range(featureSize):
        soundSimilarity += weights[i]*multiplier[i]
    return soundSimilarity
    
# 计算字形编码相似性的函数
def computeShapeCodeSimilarity(shapeCode1, shapeCode2):
    # 特征大小（形状编码的长度）
    featureSize=len(shapeCode1)
    # 特征权重
    weights=[0.15,0.15,0.15,0.15,0.15,0.15,0.1]
    multiplier=[]
    # 计算形状编码的相似性
    for i in range(featureSize):
        if shapeCode1[i]==shapeCode2[i]:
            multiplier.append(1)
        else:
            multiplier.append(0)
    shapeSimilarity=0
    # 计算形状编码的相似性
    for i in range(featureSize):
        shapeSimilarity += weights[i]*multiplier[i]
    return shapeSimilarity

# 计算字符相似性的函数
def computeSSCSimilarity(ssc1, ssc2):
    # 组合字音和字形的相似性，根据权重计算
    shapeSimi=computeShapeCodeSimilarity(ssc1[4:], ssc2[4:])
    soundSimi=computeSoundCodeSimilarity(ssc1[:4], ssc2[:4])
    return max(soundSimi, shapeSimi)


# 构建字符相似性网络（用矩阵形式表示）
def compute_sim_mat(chinese_characters, chinese_characters_code,output_file):
    sim_mat = [[0] * len(chinese_characters) for _ in range(len(chinese_characters))]
    for i in tqdm(
        range(len(chinese_characters)), desc="Constructing Similarity Matrix", unit="i"
    ):
        for j in range(i, len(chinese_characters)):
            similarity = computeSSCSimilarity(
                chinese_characters_code[chinese_characters[i]],
                chinese_characters_code[chinese_characters[j]],
            )
            sim_mat[i][j] = similarity
            sim_mat[j][i] = similarity

    # 将结果保存到pkl文件
    with open(output_file, "wb") as f:
        pickle.dump(sim_mat, f)

    return sim_mat


# 从pkl文件中加载相似性矩阵
def load_sim_mat(filename):
    with open(filename, "rb") as f:
        sim_mat = pickle.load(f)

    return sim_mat