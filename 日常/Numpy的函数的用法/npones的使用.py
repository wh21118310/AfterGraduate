import numpy as np
# zeros()返回一个全0的n维数组，共三个参数:
# shape(用来指定返回数组的大小)
# dtype(数组元素的类型)
# order(是否以内存中的C或Fortran连续顺序存储多维数据)
print(np.zeros(5))
print(np.zeros((2,2)))
#ones()返回一个全1的n维数组，有三个参数：
# shape（用来指定返回数组的大小）
# dtype（数组元素的类型）
# order（是否以内存中的C或Fortran连续（行或列）顺序存储多维数据）
print(np.ones(5))
print(np.ones((5,5),dtype=int))
