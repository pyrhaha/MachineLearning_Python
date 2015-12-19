#encoding:utf-8
import adaboost
from numpy import *
#单层决策树生成函数
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#传入的参数为数据集，选择的特征，阈值，选择的设置标签的方式
#将数据集根据阈值设置标签，然后与实际标签进行比较得出错误率
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq =='lt':#用于选择小于阈值的赋为-1还是大于阈值的赋为-1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
        #使用数组过滤的方法，其中dataMatrix[:,dimen] <= threshVal是将两个矩阵进行比较，返回一个布尔值的列表，若成立则是True，对retArray的赋值操作会将True位置的值赋为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0#表示将训练集如何切，即阈值取多少，用于在特征的所有可能值上进行遍历,比如该特征的可能取值为1,2,3,4，numSteps为3，则（4-1）/3 = 1,于是选择阈值分别为min-1，min，min+1,min+2,.....,min+numStep+1,即0,1,2,3,4,5
    bestStump = {}#这个字典用于存储给定权重向量D时所得到的最佳单层决策树
    bestClasEst = mat(zeros((m,1)))
    minError = inf#表示最小错误率，初始化为正无穷
    for i in range(n):#第一层循环，用于将所有的特征遍历
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps#表示被切割的每一部分的间隔
        for j in range(-1,int(numSteps)+1):#遍历每一种可能的阈值
            for inequal in ['lt','gt']:#遍历阈值左右两边不同的标签，比如左边是-1还是右边是-1
                threshVal = rangeMin + float(j) * stepSize#设置不同的阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#返回分类结果
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0#将预测的标签和实际标签进行比较，若相等则置为0，这样在计算
                weightErr = D.T * errArr#权重向量和错误率做为最后的错误率，这是ababoost与分类器交互的地方
                print "split dim %d, thresh %.2f ,thresh inequal:%s, the weighted error is %.3f" % (i,threshVal,inequal,weightErr)
                if weightErr < minError:
                    minError = weightErr
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

#dataMat,labelMat = adaboost.loadSimpData()
#D = mat(ones((5,1))/5)
#bestStump,minError,bestClasEst = buildStump(dataMat,labelMat,D)
#print bestStump,minError,bestClasEst