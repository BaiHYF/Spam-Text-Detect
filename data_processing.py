"""
数据预处理模块

- Author: BaiHYF <baiheyufei@gmail.com>
- Date:   Mon May 19 2025

包含以下功能：
- 数据集划分
- 汉字统计与编码生成
- 预存字符编码加载

Functions:
    divide_dataset(filename, lines=10000) -> tuple[list, list]
    count_chinese_characters(content, output_file_path) -> tuple[list, dict, dict]
    load_chinese_characters(filename) -> tuple[list, dict, dict]

大部分程序参考了 《数据科学与工程实战》王昌栋，赖剑煌 第 4.1.3 节的代码实现 
"""

from tqdm import tqdm
from character_coder import ChineseCharacterCoder

def divide_dataset(filename, lines=10000):
    """
    划分原始数据集为标签和文本

    Args:
        filename (str): 原始数据文件路径
        lines (int): 读取行数限制，默认10000

    Returns:
        tuple: (tag列表, text列表)

    Raises:
        FileNotFoundError: 当输入文件不存在时抛出
    """
    with open (filename, "r", encoding="utf-8") as f:
        text_data = f.readlines()

    subset = text_data[:lines]
    
    dataset = [s.strip().split("\t") for s in subset]
    # dataset = [data for data in dataset if len(data) == 2 and data[1].strip()]

    tag = [data[0] for data in dataset]
    text = [data[1] for data in dataset]

    print("Dataset size: ", len(dataset))
    print("Tags: ", set(tag))
    
    return tag, text

def count_chinese_characters(content, output_file_path):
    """
    统计文本中的汉字并生成编码文件

    Args:
        content (list[str]): 文本数据列表
        output_file_path (str): 输出文件路径

    Returns:
        tuple: (汉字列表, 汉字计数字典, 汉字编码字典)

    Notes:
        输出文件格式为：汉字\t编码\t出现次数
    """
    chinese_characters = []
    chinese_characters_count = {}
    chinese_characters_code = {}

    for line in tqdm(content, desc="Counting characters", unit="line"):
        for char in line:
            if "\u4e00" <= char <= "\u9fff":  # 判断是否为汉字
                chinese_characters_count[char] = (
                    chinese_characters_count.get(char, 0) + 1
                )

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for char, count in tqdm(
            chinese_characters_count.items(),
            desc="Computing Character Code",
            unit="char",
        ):
            character_code = ChineseCharacterCoder().generate_character_code(char)
            chinese_characters_code[char] = character_code
            output_file.write(f"{char}\t{character_code}\t{count}\n")
            chinese_characters.append(char)

    print(f"Results saved to {output_file_path}")

    return chinese_characters, chinese_characters_count, chinese_characters_code

def load_chinese_characters(filename):
    """
    加载预存的汉字编码数据

    Args:
        filename (str): 编码文件路径

    Returns:
        tuple: (汉字列表, 计数字典, 编码字典)

    Raises:
        FileNotFoundError: 当文件不存在时抛出
    """
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readlines()
    chinese_characters = []
    chinese_characters_count = {}
    chinese_characters_code = {}

    for row in line:
        char, code, count = row.strip().split("\t")
        chinese_characters.append(char)
        chinese_characters_code[char] = code
        chinese_characters_count[char] = count

    return chinese_characters, chinese_characters_count, chinese_characters_code
