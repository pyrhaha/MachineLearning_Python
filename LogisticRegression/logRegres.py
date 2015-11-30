#encoding:utf-8
#logistic回归梯度上升优化算法
from math import *
import matplotlib.pyplot as plt
from numpy import *
def loadDataSet():#加载数据
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()#将一行字符串中开头和结尾的空格消除，同时将其进行分隔
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#txt数据格式为前两列表示两个特征，第三列为标签，我们需要一个数据集为第一列表示w0，即偏移量，赋值为1，第二三列为特征
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
def gradAscent(dataMatIn,classLabels):#梯度下降算法
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001#学习速率
    maxCycles = 500#最大迭代周期
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)#h为估计函数，将输入特征与权值的积作为函数的参数
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error#权值更新，dataMatrix.transpose()*error为cost function对w求导的结果，具体的证明过程见浏览器收藏夹的机器学习文件夹的logistic，或机器学习实战logistic板块
    return weights

def plotBestFit(weights):#绘制图像
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    #为了拟合分隔线，0=w0*x0+w1*x1+w2*x2,其中w0为常数项，所以x0为1，x1为x，x2为y,拟合出x1和x2之间的关系
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def stocGradAscent0(dataMatrix,classLabels):#随机梯度下降算法
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights + alpha*error*dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)#生成一组按顺序的序列
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01#学习步长随着迭代的次数变小
            randIndex = int(random.uniform(0,len(dataIndex)))#书上的做法：其实是将dataIndex作为记录剩余样本数的计数器
            #randIndex = dataIndex[int(random.uniform(0,len(dataIndex)))]#我的做法：将dataIndex生成的0-m的序列，从中随机取出样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * dataMatrix[randIndex] * error
            #dataIndex.remove(randIndex)
            del dataIndex[randIndex]
    return weights

def classifyVector(features,weights):#用于分类的函数，输入参数为测试集的特征以及训练出来的模型中的权值
    prob = sigmoid(sum(features*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():#使用训练集训练模型并用测试集测试模型
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(trainingSet,trainingLabels,500)
    #得到训练模型的结果
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        result = classifyVector(array(lineArr),trainWeights)
        if int(result) != int(currLine[21]):
            errorCount+=1
    errorRate = (float(errorCount)/numTestVec)
    print 'the error rate of the test is: %f' % errorRate
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print 'after %d iterations the average error rate is: %f '%(numTests,errorSum/float(numTests))





#dataArr,labelMat = loadDataSet()
#weights = stocGradAscent1(dataArr,labelMat,500)
#plotBestFit(weights)
#plotBestFit(weights.getA())#返回的权值是矩阵时需要通过getA返回array形式
multiTest()


