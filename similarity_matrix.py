"""
相似度矩阵模块

- Author: BaiHYF <baiheyufei@gmail.com>
- Date:   Mon May 19 2025

包含相似度矩阵的以下操作：
- 矩阵计算与存储
- 矩阵加载
- 矩阵动态更新

Functions:
    compute_sim_mat(chinese_characters, chinese_characters_code) -> list[list[float]]
    load_sim_mat(filename) -> numpy.ndarray
    update_sim_mat(new_characters, chinese_characters_code, sim_mat) -> numpy.ndarray

大部分程序参考了 《数据科学与工程实战》王昌栋，赖剑煌 第 4.3.1 节的代码实现     
"""


import numpy as np
from tqdm import tqdm
import pickle
from ssc_similarity import computeSSCSimilarity

# 构建字符相似性网络（用矩阵形式表示）
def compute_sim_mat(chinese_characters, chinese_characters_code):
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
    output_file = "Data/similarity_matrix.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(sim_mat, f)

    return sim_mat


# 从pkl文件中加载相似性矩阵
def load_sim_mat(filename):
    with open(filename, "rb") as f:
        sim_mat = pickle.load(f)

    return sim_mat


# 更新相似性矩阵
def update_sim_mat(new_characters, chinese_characters_code, sim_mat):
    for char in new_characters:
        # 计算新汉字与现有汉字之间的相似性
        new_code = chinese_characters_code[char]
        similarities = [
            computeSSCSimilarity(new_code, code)
            for code in chinese_characters_code.values()
        ]

        # 更新相似性矩阵
        new_row = np.array(similarities)
        sim_mat = np.vstack([sim_mat, new_row])
        sim_mat = np.hstack([sim_mat, new_row.reshape(-1, 1)])

    return sim_mat