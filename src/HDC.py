import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def generateDDR(D):
    randomHV=np.ones(D)
    permuIdx=np.random.permutation(D)           # 生成了一个长度为 D 的随机排列索引数组 permuIdx
    randomHV[permuIdx[:int(D/2)]] = -1          # 将 randomHV 数组中一半的元素（对应随机索引 permuIdx 的前半部分）修改为 -1
    # for i in range(int(len(permuIdx)/2)):
    #     randomHV[permuIdx[i]]=-1
    randomHV=randomHV.astype(int)               # 将 randomHV 数组的类型转换为 int，确保所有元素都是整数型的 1 或 -1
    return randomHV

# def binarizeMajorityRule(HV):
#     hv = np.ones(len(HV), dtype='float32')
#     for i in range(len(HV)):
#         if HV[i] < 0:
#             hv[i] = -1
#     return hv

def binarizeMajorityRule(HV):
    # 通过 np.sign 更高效地进行二值化处理
    return np.sign(HV)

# def encoding(sampleX,D,pixelMemory,valueMemory):
#     sampleHV=np.zeros(D, dtype='float32')
#     num_pixels=len(sampleX)
#     for i in range(num_pixels):
#         sampleHV += pixelMemory[i]*valueMemory[sampleX[i]]
#     # return sampleHV                             # Non-binary
#     return binarizeMajorityRule(sampleHV)     # Binary

def encoding_vectorized(sampleX, D, pixelMemory, valueMemory):
    # 使用 NumPy 向量化操作代替逐元素循环
    sampleHV = np.sum(pixelMemory[:len(sampleX)] * valueMemory[sampleX], axis=0)
    return binarizeMajorityRule(sampleHV)

def batch_encoding(x_train, dimension, featureMemory, valueMemory):
    # 使用 joblib 并行化编码
    return Parallel(n_jobs=-1)(
        delayed(encoding_vectorized)(x, dimension, featureMemory, valueMemory) for x in x_train
    )

def inference(y_test, test_HVs, associativeMemory, num_class):
    num_class = num_class
    length = len(test_HVs)
    all_list = np.zeros(num_class)
    correct_list = np.zeros(num_class)

    for i in range(length):
        hamm_list = []
        Y = test_HVs[i]
        inds = []
        for j in range(num_class):
            item  = np.dot(Y, associativeMemory[j])     # 通过点积计算当前测试样本与类别 j 的类超向量的相似度。
            hamm_list.append(item)
        pred_class = np.argmax(hamm_list)
        all_list[int(y_test[i])] = all_list[int(y_test[i])] + 1
        if pred_class == y_test[i]:
            correct_list[int(y_test[i])] = correct_list[int(y_test[i])] + 1

    sum_corr = 0
    sum_all = len(y_test)
    for i in range(num_class):
        sum_corr += correct_list[i]
    print('Avg: %.4f'%(sum_corr/sum_all))
    return sum_corr/sum_all
