# encoding:utf-8
# @Author: DorisFawkes
# @File:
# @Date: 2021/08/03 17:04

#1.random.random()生成0~1以内的随机数
#2.random.seed(x)随机种子，x可以是任意数字
# 若设定x，则每次调用生成的随机数会是同一个
# import random
# a = random.random()
# print(a)
# random.seed()
# b = random.random()
# print(b)
# random.seed(0)
# c = random.random()
# print(c)
# random.seed(0)
# d = random.random()
# print(d)
# random.seed(1)
# e = random.random()
# print(e)
import os
path = './data/src'
if not os.path.exists(path):
    print("false")
    os.makedirs(path)