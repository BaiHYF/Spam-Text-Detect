"""
汉字编码器模块

- Author: BaiHYF <baiheyufei@gmail.com>
- Date:   Mon May 19 2025

包含汉字编码生成功能，支持：
- 拼音编码生成
- 字形编码生成
- 完整字符编码组合

Classes:
    ChineseCharacterCoder

大部分程序参考了 《数据科学与工程实战》王昌栋，赖剑煌 第 4.3.1 节的代码实现  
"""

from pypinyin import pinyin, Style
from four_corner_method import FourCornerMethod

class ChineseCharacterCoder:
    def __init__(self):
        # 初始化字典
        self.structure_dict = {}
        self.strokes_dict = {
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
            "10": "A",
            "11": "B",
            "12": "C",
            "13": "D",
            "14": "E",
            "15": "F",
            "16": "G",
            "17": "H",
            "18": "I",
            "19": "J",
            "20": "K",
            "21": "L",
            "22": "M",
            "23": "N",
            "24": "O",
            "25": "P",
            "26": "Q",
            "27": "R",
            "28": "S",
            "29": "T",
            "30": "U",
            "31": "V",
            "32": "W",
            "33": "X",
            "34": "Y",
            "35": "Z",
            "36": "a",
            "37": "b",
            "38": "c",
            "39": "d",
            "40": "e",
            "41": "f",
            "42": "g",
            "43": "h",
            "44": "i",
            "45": "j",
            "46": "k",
            "47": "l",
            "48": "m",
            "49": "n",
            "50": "o",
            "51": "p",
        }

        # 加载汉字结构对照文件
        with open("Data/hanzijiegou_2w.txt", "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    structure, chinese_character = parts
                    self.structure_dict[chinese_character] = structure

        # 加载汉字笔画对照文件，参考同级目录下的 chinese_unicode_table.txt 文件格式
        self.chinese_char_map = {}
        with open("Data/chinese_unicode_table.txt", "r", encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines[6:]:  # 前6行是表头，去掉
                line_info = line.strip().split()
                # 处理后的数组第一个是文字，第7个是笔画数量
                self.chinese_char_map[line_info[0]] = self.strokes_dict[line_info[6]]

    def split_pinyin(self, chinese_character):
        # 将汉字转换为拼音（带声调）
        pinyin_result = pinyin(chinese_character, style=Style.TONE3, heteronym=True)

        # 多音字的话，选择第一个拼音
        if pinyin_result:
            py = pinyin_result[0][0]

            initials = ""  # 声母
            finals = ""  # 韵母
            codas = ""  # 补码
            tone = ""  # 声调

            # 声母列表
            initials_list = [
                "b",
                "p",
                "m",
                "f",
                "d",
                "t",
                "n",
                "l",
                "g",
                "k",
                "h",
                "j",
                "q",
                "x",
                "zh",
                "ch",
                "sh",
                "r",
                "z",
                "c",
                "s",
                "y",
                "w",
            ]

            # 韵母列表
            finals_list = [
                "a",
                "o",
                "e",
                "i",
                "u",
                "v",
                "ai",
                "ei",
                "ui",
                "ao",
                "ou",
                "iu",
                "ie",
                "ve",
                "er",
                "an",
                "en",
                "in",
                "un",
                "vn",
                "ang",
                "eng",
                "ing",
                "ong",
            ]

            # 获取声调
            if py[-1].isdigit():
                tone = py[-1]
                py = py[:-1]

            # 获取声母
            for initial in initials_list:
                if py.startswith(initial):
                    initials = initial
                    py = py[len(initial) :]
                    break

            # 获取韵母
            for final in finals_list:
                if py.endswith(final):
                    finals = final
                    py = py[: -len(final)]
                    break

            # 获取补码
            codas = py

            return initials, finals, codas, tone

        return None

    def generate_pronunciation_code(self, hanzi):
        initial, final, coda, tone = self.split_pinyin(hanzi)

        # 轻声字，例如'了'
        if tone == "":
            tone = "0"

        # 声母映射
        initials_mapping = {
            "b": "1",
            "p": "2",
            "m": "3",
            "f": "4",
            "d": "5",
            "t": "6",
            "n": "7",
            "l": "8",
            "g": "9",
            "k": "a",
            "h": "b",
            "j": "c",
            "q": "d",
            "x": "e",
            "zh": "f",
            "ch": "g",
            "sh": "h",
            "r": "i",
            "z": "j",
            "c": "k",
            "s": "l",
            "y": "m",
            "w": "n",
        }

        # 韵母映射
        finals_mapping = {
            "a": "1",
            "o": "2",
            "e": "3",
            "i": "4",
            "u": "5",
            "v": "6",
            "ai": "7",
            "ei": "8",
            "ui": "9",
            "ao": "a",
            "ou": "b",
            "iu": "c",
            "ie": "d",
            "ve": "e",
            "er": "f",
            "an": "g",
            "en": "h",
            "in": "i",
            "un": "j",
            "vn": "k",
            "ang": "l",
            "eng": "m",
            "ing": "n",
            "ong": "o",
        }

        # 补码映射
        coda_mapping = {"": "0", "u": "1", "i": "1"}

        # 获取映射值
        initial_code = initials_mapping.get(initial, "0")
        final_code = finals_mapping.get(final, "0")
        coda_code = coda_mapping.get(coda, "0")

        # 组合生成四位数的字音编码
        pronunciation_code = initial_code + final_code + coda_code + tone

        return pronunciation_code

    def generate_glyph_code(self, hanzi):
        # 获取汉字的结构
        structure_code = self.structure_dict[hanzi]

        # 获取汉字的四角编码
        fcc = FourCornerMethod().query(hanzi)

        # 获取汉字的笔画数
        stroke = self.chinese_char_map[hanzi]

        # 组合生成的字形编码
        glyph_code = structure_code + fcc + stroke

        return glyph_code

    def generate_character_code(self, hanzi):
        return self.generate_pronunciation_code(hanzi) + self.generate_glyph_code(hanzi)

