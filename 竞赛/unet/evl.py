import numpy as np
import glob
import cv2


"""
confusionMetric
P\L     P    N

P      TP    FP

N      FN    TN

a
"""
def fast_hist(imgPredict, imgLabel, numClass):
    mask = (imgLabel >= 0) & (imgLabel < numClass)
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass,numClass)
    return  confusionMatrix


def Compute_FWIou(Predict,Label,numclass):
    hist = np.zeros((numclass, numclass))
    #将所有样本的混淆矩阵相加
    for i in range(len(Predict)):
        predict=Predict[i]
        label=Label[i]
        hist += fast_hist(predict, label, 2)

    # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
    freq = np.sum(hist, axis=1) / np.sum(hist)
    iu = np.diag(hist) / (
            np.sum(hist, axis=1) + np.sum(hist, axis=0) -
            np.diag(hist))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def Compute_pixelAccuracy(Predict,Label,numclass):
    hist = np.zeros((numclass, numclass))
    # 将所有样本的混淆矩阵相加
    for i in range(len(Predict)):
        predict = Predict[i]
        label = Label[i]
        hist += fast_hist(predict, label, 2)
    # acc = (TP + TN) / (TP + TN + FP + TN)
    acc = np.diag(hist).sum() / hist.sum()
    return acc

def classPixelAccuracy(Predict,Label,numclass):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
    hist = np.zeros((numclass, numclass))
        # 将所有样本的混淆矩阵相加
    for i in range(len(Predict)):
            predict = Predict[i]
            label = Label[i]
            hist += fast_hist(predict, label, 2)
    classAcc = np.diag(hist) / hist.sum(axis=1)
    return classAcc

def meanPixelAccuracy(Predict,Label,numclass):
    hist = np.zeros((numclass, numclass))
        # 将所有样本的混淆矩阵相加
    for i in range(len(Predict)):
            predict = Predict[i]
            label = Label[i]
            hist += fast_hist(predict, label, 2)

    classAcc = np.diag(hist) / hist.sum(axis=1)
    meanAcc = np.nanmean(classAcc)
    return meanAcc
if __name__ == '__main__':
    tests_path = glob.glob('sardata/test/predict/*.png')
    Predict=[]
    Label=[]
    # 遍历素有图片
    for test_path in tests_path:
        img = cv2.imread(test_path)
        # 类别划分为0和1
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(img.shape[0]*img.shape[1])
        img[img==255]=1
        # 转为batch为1，通道为1，大小为512*512的数组
        Predict.append(img)
    tests_path = glob.glob('sardata/test/label/*.png')
    for test_path in tests_path:
        img = cv2.imread(test_path)
        # 类别划分为0和1
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(img.shape[0]*img.shape[1])
        img[img == 255] = 1
        # 转为batch为1，通道为1，大小为512*512的数组
        Label.append(img)

    FWIou=Compute_FWIou(Predict,Label,2)
    Acc=Compute_pixelAccuracy(Predict,Label,2)
    PA=classPixelAccuracy(Predict,Label,2)
    MPA=meanPixelAccuracy(Predict,Label,2)
    print("单样本分类精度:")
    print("背景:",PA[0])
    print("养殖场",PA[1])
    #print("")
    print("总体准确率：")
    print("PA=",Acc)
    #print("")
    print("类别像素平均准确率：")
    print("MPA=",MPA)
    #print("")
    print("频率权值并交比：")
    print("FWIOU=",FWIou)


