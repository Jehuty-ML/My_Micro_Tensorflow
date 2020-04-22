

import os
import numpy as np
import tensorflow as tf
# https://www.qqxiuzi.cn/zh/hanzi-gb2312-bianma.php


def chinese_to_index(text):
    rezult = []
    bs = text.encode('gb2312')  # 转换成了字节bytes格式
    bs_list = [b for b in bs]
    # print(bs_list)
    num = len(bs_list)
    i = 0
    while i < num:
        b = bs[i]
        # 如果取值小于160，表示是单个的字符（这里很特殊）
        if b <=160:
            rezult.append(b)
        else:
            # 计算区码（大于160的先计算区码）

            block = b - 160
            if block >=16:
                # 因为10--15区为空，不需要考虑
                block -= 6

            # 计算在当前区有多少汉字。（计算位置码）
            block -= 1
            i += 1
            b2 = bs[i]

            # 基于区码+位置码 计算出1个数字（这个数字就是 tokenize）
            rezult.append(block*94 + b2)
        i += 1
    return rezult

def index_to_chinese(index_list):
    """
    功能：{整数：单词}
    :param index_list:
    :return:
    """
    result = ''
    for index in index_list:
        if index <= 160:
            result += chr(index)
        else:
            index = index - 161
            block = int(index / 94) +1
            if block >= 10:
                block += 6
            block += 160
            # 位置码
            location = int(index % 94) + 161
            result += str(bytes([block, location]), encoding='gb2312')
    return result


def read_poems(path='../qts_7X4.txt'):
    """
    读入唐诗数据
    :param path:
    :return:
    """
    rezult = []
    error = 0
    with open(path, mode='r', encoding='utf-8') as reader:
        for line in reader:
            # 对每行数据进行处理，去除掉前后空格。
            line = line.strip()
            length = len(line)
            try:
                if length == 32:
                    index = chinese_to_index(line)
                    rezult.append(index)
                else:
                    error += 1
            except:
                error += 1
    print('成功获取诗歌:{} - 错误:{}'.format(len(rezult), error))
    return rezult


def fetch_samples(path='../qts_7X4.txt'):
    """
    基于原始数据构建 X和Y
    :param path:
    :return:
    """
    total_samples = 0
    X = read_poems(path)
    Y = []
    for xi in X:
        total_samples +=1
        # 使用前一个字预测下一个字。
        yi = xi[1:]
        yi.append(10)  # 目标yi最后会少1个字符，用10代替。
        # 添加到Y的列表当中。
        Y.append(yi)
    # 将X 和 Y 转换成为 numpy 数组
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(X.shape, Y.shape)
    return total_samples, X, Y

# text = '澄潭皎镜石崔巍.万壑千岩暗绿苔.林亭自有幽贞趣.况复秋深爽气来.'
# sentence = chinese_to_index(text)
# test_new = index_to_chinese(sentence)
# print(sentence)
# print(test_new)

# x = read_poems(path='../qts_7X4.txt')
total_samples, X, Y = fetch_samples(path='../qts_7X4.txt')