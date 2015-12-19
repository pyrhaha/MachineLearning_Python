#encoding:utf-8
from numpy import *
import matplotlib.pyplot as plt
#k-均值函数的一些支持函数
def loadDataSet(filename):#读取数据集
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)#将curLine中的元素映射到float中，该函数的功能相当于将curLine中的每一个元素使用float函数进行处理
        dataMat.append(fltLine)
    return dataMat

#距离函数，这里使用的是欧氏距离，也可以换成其他距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#该函数生成k个随机质心
def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#用于存储生成的所有随机质心，为行数为k，列数为特征的个数
    for j in range(n):
        maxJ = float(max(dataSet[:,j]))
        minJ = float(min(dataSet[:,j]))
        rangeJ = maxJ - minJ
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)#random.rand函数生成k行0~1之间的随机数,使得质心的范围在min到max之间
    return centroids

#k-均值聚类算法
def kMeans(dataSet,k,distMeas = distEclud,createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(ones((m,2)))#用于存储每一个数据集的属于类别信息，第一列表示属于的簇类别，第二列表示误差，即当前点到质心的距离
    centroids  = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:#若质心有发生变化则继续循环，直到质心不再变化
        clusterChanged = False
        for i in range(m):#每次循环遍历所有的训练集
            minDist = inf
            minIndex = -1
            for j in range(k):#尝试每一个簇类
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:#如果存在一个训练集的所属簇类发生变化的话就将clusterChanged设为True
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0] == cent)[0]][0]#使用数组过滤提取出属于每一个簇类的数据集
            centroids[cent,:] = mean(ptsInClust,axis=0)#axis等于0表示对一列求平均值，即对一个特征的所有取值求平均值
    return centroids,clusterAssment#centroids存储的是不同簇类的特征值质心的取值，每一列表示一个特征，每一行表示一个簇类，clusterAssment存储的是所有训练集的所属簇类（第一列）和其与质心之间的误差（第二列）





dataMat = mat(loadDataSet('testSet.txt'))
#print randCent(dataMat,2)
#print disEclud(dataMat[0],dataMat[1])
myCentroids,clusterAssing = kMeans(dataMat,4)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0],dataMat[:,1])#画点
ax.scatter(myCentroids[:,0],myCentroids[:,1],c = 'red')
plt.show()