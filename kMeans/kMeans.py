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

#二分K均值聚类算法，是指从一个簇类开始，每一次将一个簇类分成两个，减小误差，直到达到k个聚类
def binaryKMeans(dataSet,k,distMeans = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#和上一个函数一样用于存储每个数据集的所属簇类和误差
    centroid0 = mean(dataSet,axis=0).tolist()[0]#初始化簇类，一开始只有一个簇类，所以质点就是所有训练集各特征的平均值
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeans(mat(centroid0),dataSet[j,:])**2#计算这时的误差
    while (len(centList) < k):#不断循环直到达到k个簇类
        lowestSSE = inf
        for i in range(len(centList)):
            childDataSet = dataSet[nonzero(clusterAssment[:,0]== i)[0]][0]#获取数据集中属于该簇类的数据集
            centroidMat,splitCluster = kMeans(childDataSet,2,distMeans)#将一个簇类分成两个簇类
            errorSplit = float(sum(splitCluster[:,1]))#将重新分簇类的数据集中误差求和
            errorNoSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])#为已正确分簇的数据集求误差和
            print "the error of split is ",errorSplit,",error of nonsplit is ",errorNoSplit
            if (errorNoSplit + errorSplit) < lowestSSE:#如果其他簇的误差和该簇分开后的簇的和比最小的误差要小则将其赋值为最小的误差
                bestCentSplit = i#此时i为最好的需要切分的簇
                bestNewCent = centroidMat#存储此时切分后的两个簇的质点
                bestDataCluster = splitCluster.copy()#存储簇被切分的那些数据集此时的簇索引以及误差
                lowestSSE = errorNoSplit + errorSplit
        bestDataCluster[nonzero(bestDataCluster[:,0] == 0)[0],0] = bestCentSplit#将被切分簇里的数据集中所属的簇进行修改，因为这是讲一个簇分成两个簇，所以将0簇赋值为原来未分之前簇的索引
        bestDataCluster[nonzero(bestDataCluster[:,0] == 1)[0],0] = len(centList)#将1簇增加为len(centList)，因为之前的编号是0~len(centList)-1，所以相当于新加了一个
        print "the best cluster to split is ",float(bestCentSplit)
        print "the len of dataSet to be split is ",float(len(bestDataCluster))
        centList[bestCentSplit] = bestNewCent[0,:].tolist()[0]#将原来未切分的簇替换成切分后的第一个簇
        centList.append(bestNewCent[1,:].tolist()[0])#将切分后的第二个簇加入centList
        clusterAssment[nonzero(clusterAssment[:,0] == bestCentSplit)[0]] = bestDataCluster
    return mat(centList),clusterAssment


# dataMat = mat(loadDataSet('testSet.txt'))
# #print randCent(dataMat,2)
# #print disEclud(dataMat[0],dataMat[1])
# myCentroids,clusterAssing = kMeans(dataMat,4)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0],dataMat[:,1])#画点
# ax.scatter(myCentroids[:,0],myCentroids[:,1],c = 'red')
# plt.show()
dataMat = mat(loadDataSet('testSet2.txt'))
centList,myNewAssments = binaryKMeans(dataMat,3)
print centList
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0],dataMat[:,1])#画点
ax.scatter(centList[:,0],centList[:,1],c = 'red')
plt.show()