#coding:utf-8
#Knn算法通过计算数据集和每个训练集的距离，取前k个距离最小的训练集的类标签，判断该数据集的类标签
#如：数据集为[x1,y1], 有一个训练集[x0,y0], 则计算距离为(x1-x0)^2+(y1-y0)^2
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
def createDataSet():#创建数据集
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX,dataSet,labels,k):#分类算法实现，inX表示用于分类的输入向量，dataSet表示训练集，labels表示标签信息，k表示获取前k个最相近的数据集
    dataSetSize = dataSet.shape[0]#shape函数得到的是一个(行，列)的元组

    diffMat = tile(inX,(dataSetSize,1)) - dataSet#将输入向量扩充为和训练集行数一样的矩阵，使其能和每一个训练集求差
    #tile函数的功能是扩充矩阵,tile(需要扩充的矩阵,一个表示扩充行列倍数的元组),比如原来的矩阵为[[1,2],[2,3]]元组为(2,3),则表示矩阵的行扩充为原来的2倍,列扩充为原来的3倍
    #结果为[[1,2,1,2,1,2],[2,3,2,3,2,3],[1,2,1,2,1,2],[2,3,2,3,2,3]],若是tile的第二个参数只有一个数字3，则默认为(1,3)

    sqDiffMat = diffMat **2
    sqDistances = sqDiffMat.sum(axis=1)
    #sum函数是用于求和，若axis参数为空，则将所有的数字相加，得到的是一个数，若axis=0，则将各行相加，比如sum([[0, 1], [0, 5]], axis=0)，结果为[0,6].若axis=1，则将各列相加，并转为数组,比如sum([[0, 1], [1, 5]], axis=1)，结果为[1,6]

    distances = sqDistances**0.5#计算距离的公式为((x1-x2)平方+(y1-y2)平方)开根号
    sortedDistIndicies = distances.argsort()#argsort函数的功能是返回数组从小到大排列的索引值，比如argsort([3, 1, 2])，得到的结果为[1, 2, 0]
    classCount = {}#该字典对应不同标签的投票值，即在排序前k个训练集中有多少个是该标签
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]#获取前i个最小距离的点对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1#若该输入向量和某一训练集的距离在前k个，则该训练集对应的标签在字典中的值加1

    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    #对字典的数据进行排序，返回值是排好序的列表，sorted函数的第一个参数是一个迭代器，第二个参数是用于确定采用什么标准排序，第三个参数是True表示从大到小的排序
    #字典的iteritems函数是python2中用于获取该字典的迭代器的方法，python3中为items函数。operator.itemgetter(1)是指将迭代器中第二个域元素的作为排序的标准，由于字典的迭代器的元素形式为如[('A',1),('B',2)]，是一个列表,所以就是采用值作为标准

    return sortedClassCount[0][0]#返回值最大的对应的标签，即分类结果

def file2matrix(filename):#将约会数据从文件中格式化变成矩阵
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)#得到文件的行数
    returnMat = zeros((numberOfLines,3))#创建全零矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()#将每一行开头和结尾的空格清空掉
        listFromLine = line.split('\t')#用\t分割字符串
        returnMat[index,:] = listFromLine[0:3]#每一行记录一组数据集，三个特征
        classLabelVector.append(listFromLine[-1])#将标签存入classLabelVector列表中
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):#归一化公式为(oldValue-min)/(max-min)
    minVals = dataSet.min(0)#参数为0表示获取矩阵每一列的最小值，若参数为1表示获取每一行最小值
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))#创建一个和原数据集一样大小的零矩阵
    m = dataSet.shape[0]#获取行数
    normDataSet = dataSet - tile(minVals,(m,1))#将每一个数据集都和最小值求差
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals


def datingClassTest():#用于测试该分类器的准确度
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)#设置测试数据集的数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],3)
        #取前10%的数据集作为测试集，后面的数据作为训练集
        print classifierResult+","+datingLabels[i]
        if (classifierResult != datingLabels[i]):
            errorCount+=1.0

    print "the Total error rate is: %f" % (errorCount/float(numTestVecs))

def img2vector(filename):#将32*32的图像转换为[1,1024]的向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):#循环读出文件的前32行
        lineStr = fr.readline()
        for j in range(32):#每行的头32个字符
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():#手写数字识别系统的测试
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
        #该矩阵的每一行存储一个图像

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mtest = len(testFileList)
    for i in range(mtest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        if (classifierResult != classNumStr):
            errorCount+=1.0
    print "the Total error rate is: %f" % (errorCount/float(mtest))

#datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
#plt.show()
#normMat,ranges,minVals = autoNorm(datingDataMat)
#print normMat
#print ranges
#print minVals
#datingClassTest()
#testVector = img2vector('testDigits/0_13.txt')
#print testVector[0,0:31]
handwritingClassTest()
