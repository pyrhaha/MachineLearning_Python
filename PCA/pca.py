#encoding:utf-8
#该算法介绍的是PCA，主成分分析，用于降维。实现原理是将原数据投影到k维的新坐标上，选择方差最大的前k个互相正交的直线方向作为坐标轴，实现过程为首先求数据各个特征之间的协方差矩阵，然后求其特征值和特征向量
#因为要选择方差最大的前k个方向就是求前k个最大的特征值，其对应的特征向量就是要求的方向向量
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
#读取数据
def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,data) for data in stringArr]
    return mat(datArr)

#pca算法实现
def pca(dataMat,TopFeatureNumber=999999):#该函数的输入参数为数据集和降维后的维度，默认很大
    Datamean = dataMat.mean(axis=0)#求每列的平均值
    standardData = dataMat - Datamean#将数据进行标准化，使其的平均值变为0
    covMat = cov(standardData.T)#求协方差
    featureVal,featureMat = linalg.eig(covMat)#求特征值和特征向量，其中特征值featureVal[i]对应特征向量featureMat[:,i]
    featSortInd = argsort(-featureVal)#进行降序排序
    featSortInd = featSortInd[:TopFeatureNumber]#取特征值最大的前k维
    TopFeatMat = featureMat[:,featSortInd]#获得对应的特征向量,这个矩阵的行数是原数据的维度，列数是降维后的维度
    TransData = standardData*TopFeatMat#将原数据投影到新的空间中

    reconData = (TransData*TopFeatMat.T)+Datamean#将投影后的数据转换回原来的坐标系显示，即投影后的结果
    return TransData,reconData

#在测试实际数据中存在缺失值NaN，该函数用于将这些值替换成该特征的平均值
def replaceNanWithMean():
    dataMat = loadDataSet('secom.data',' ')
    numFeat = shape(dataMat)[1]#获取特征的个数
    for i in range(numFeat):#将缺失值NaN变为该特征所有取值的平均值
        featMean = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i])
        dataMat[nonzero(isnan(dataMat[:,i])),i] = featMean
    return dataMat

#用于查看各特征值(即方差)所占的比例，为今后降维设置特征个数提供指导
def pcaTest(dataMat,TopFeatureNum=999999):#该函数的输入参数为数据集
    Datamean = dataMat.mean(axis=0)#求每列的平均值
    standardData = dataMat - Datamean#将数据进行标准化，使其的平均值变为0
    covMat = cov(standardData.T)#求协方差
    featureVal,featureMat = linalg.eig(covMat)#求特征值和特征向量，其中特征值featureVal[i]对应特征向量featureMat[:,i]
    FeatSortInd = argsort(-featureVal)#对特征值进行排序，这样图表比较好观察
    varSum = sum(featureVal)#计算特征值的总和
    percent = (featureVal[FeatSortInd]/varSum)*100
    Locator = MultipleLocator(5)#设置刻度
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = arange(0,TopFeatureNum,1)
    y = percent[:TopFeatureNum]

    ax.scatter(x,y,marker = 'o',s=50,c='red')
    ax.plot(x,y)
    ax.xaxis.set_major_locator(Locator)
    plt.show()


#dataMat = loadDataSet('testSet.txt')
# resultData,conData = pca(dataMat,1)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0],dataMat[:,1],marker = '^',s=90)
# ax.scatter(conData[:,0],conData[:,1],marker = 'o',s=50,c='red')
# plt.show()
dataMat = replaceNanWithMean()
pcaTest(dataMat,20)