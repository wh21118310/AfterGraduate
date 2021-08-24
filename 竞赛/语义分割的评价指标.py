# encoding:utf-8
# @Author: DorisFawkes
# @File:语义分割的评价指标
# @Date: 2021/08/19 21:08
import cv2
import numpy as np
__all__=['SegmentationMetric']
'''
confusionMatrix
Label\Predict   P    N
P               TP   FN
N               FP   TN
'''
class SegmentationMetric(object):
    def __init__(self,numClass):
        self.numClass = numClass
        #创建空矩阵存储混淆矩阵
        self.confusionMatrix = np.zeros((self.numClass,)*2)
        #*2相当于平方，即结果为numClass*numClass的混淆矩阵
    # def pixelAccuracy(self):
    #     # 求准确率，即预测对的占总数的比例
    #     # return all class overall pixel accuracy
    #     #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
    #     # np.diag(array) 若array是一维数组，则输出以一维数组为对角线元素的矩阵
    #     # 若array是二维矩阵，则暑促矩阵的对角线元素
    #     acc = np.diag(self.confusionMatrix).sum()/ np.sum(self.confusionMatrix)
    #     return acc
    #
    # def classPixelAccuracy(self):
    #     # 求精确率，即各类预测结果中预测对的占预测结果的比例
    #     # return each category pixel accuracy
    #     # acc = (TP) / TP + FP 或者 acc = (TN) / TN+FN
    #     classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
    #     return classAcc #返回列表，如:[0.90, 0.80, 0.96]，表示类别1、2、3的预测准确率
    #
    # def meanPixelAccuracy(self):
    #     """
    #     Mean Pixel Accuracy(MPA,平均像素精度):PA的一种简单提升，计算每类被正确分类像素数的比例，之后求所有类的平均
    #     Returns
    #     """
    #     # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
    #     meanAcc = np.nanmean(self.classPixelAccuracy())
    #     return meanAcc
    #     # 返回单值
    #     # 如：np.nanmean([0.90,0.80,0.96,nan,nan])=(0.90 + 0.80 + 0.96/3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection =  TP ,Union = TP+FP+FN, IoU = TP /(TP+FP+FN)，求交并比
        # 相当于对角线元素与对角线元素所处行列总和的比值
        intersection = np.diag(self.confusionMatrix) #取对角线元素，返回列表
        union = np.sum(self.confusionMatrix,axis=1)+np.sum(self.confusionMatrix,axis=0) - np.diag(self.confusionMatrix)
        # axis =1 表示混淆矩阵行值，返回列表;aixs=0表示列值，返回列表
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        # 求MIoU
        mIoU = np.nanmean(self.IntersectionOverUnion()) #求各类IoU的平均
        return mIoU

    def FrequencyWeightedIntersectionOverUnion(self):
        """
        FWIoU，频权交并比：根据每个类出现的频率为其设置权重
        FWIoU = [(TP + FN) / (TP + FP + TN + FN)]*[TP / (TP + FP + FN)]
        TF+FN为每个类别的真实像素量，除总像素量得到每个类别的权重
        """
        freq = np.sum(self.confusionMatrix,axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (np.sum(self.confusionMatrix,axis=1)+np.sum(self.confusionMatrix,axis=0)-np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def genConfusionMatrix(self,imgPredict,imgLabel):
        """
        计算混淆矩阵
        Parameters
            imgPredict
            imgLabel
        Returns:混淆矩阵
        """
        # ground truth(真实值)中所有正确(介于[0,numClass])的像素label的mask
        # 去除标签图中白色的轮廓，imgLabel>=0防止bincount()函数出错
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype('int') + imgPredict[mask]
        # np.bincount(x,weights=None,minlength) 计算x中每个元素的出现次数，
        # 默认返回长度比x中最大值多1的数组
        # weights表示权重数组，相当于out[n]+=weight[i],n为值，i为n的对应位置
        # minlength 表示输出数组最短长度，self.numClass **2 相当于将numClass长度平方
        # 该函数计算了从0到n^2-1这n^2个数中每个数出现的次数
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self,imgPredict,imgLabel):
        assert imgPredict.shape == imgLabel.shape
        # 得到混淆矩阵
        self.confusionMatrix += self.genConfusionMatrix(imgPredict,imgLabel)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass,self.numClass))

if __name__ == '__main__':
    imgPredict = cv2.imread("../日常/data/predict3.png")
    imgLabel = cv2.imread("../日常/data/label3.png")
    imgPredict = np.array(cv2.cvtColor(imgPredict,cv2.COLOR_BGR2GRAY) / 255.,dtype= np.uint8)
    imgLabel = np.array(cv2.cvtColor(imgLabel,cv2.COLOR_BGR2GRAY) / 255.,dtype=np.uint8)
    metric = SegmentationMetric(2)
    hist = metric.addBatch(imgPredict,imgLabel)
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    FWIoU = metric.FrequencyWeightedIntersectionOverUnion()
    print('hist is :\n', hist)
    print('IoU is : ', IoU)
    print('mIoU is : ', mIoU)
    print('FWIoU is : ',FWIoU)