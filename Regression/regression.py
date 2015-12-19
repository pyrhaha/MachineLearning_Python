#coding:utf-8
#使用简单线性回归拟合直线进行预测
from numpy import *
import matplotlib.pyplot as plt
#读取数据的函数和adaboost的一样
def loadDataSet(filename):#读取数据文件，适用于所有数据集
    numFeat = len(open(filename).readline().split('\t'))
    #读取该文件的一行，使用\t进行分隔后元素的个数就是特征的个数
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        content = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(content[i]))
        dataMat.append(lineArr)
        labelMat.append(float(content[-1]))
    return dataMat,labelMat

#采用最小二乘法来计算权值w，其中误差使用平方误差，对w求导以后公式为w = 1/(xT*X) *　xT * y
def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat#求出x的转置乘x的值
    if linalg.det(xTx) == 0.0:#该库函数是用于求矩阵的行列式，若行列式为0则矩阵没有逆，对w的求导公式就不成立
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def drawPic(xArr,yArr,ws):#绘制回归拟合的直线
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat*ws
    print corrcoef(yHat.T,yMat)#计算相关系数，参数两个都需要是行向量
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = xMat[:,1]
    y = yMat.T[:,0]
    ax.scatter(x,y)#画点
    xCopy = xMat.copy()
    xCopy.sort(0)#因为在绘图过程中如果取点不按顺序会报错
    yHat = xCopy*ws
    ax.plot(xCopy[:,1],yHat)#画线
    plt.show()

#局部加权线性回归，该方法在对测试集进行预测的时候会将离训练集较近的点的权值提高，这样对输入的测试集进行拟合的时候会主要依靠离他很近的点
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))#每一个训练集都有一个权值，初始化为创建一个m*m的单位矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp( (diffMat * diffMat.T) / (-2.0 * k**2 ))
    xTx = xMat.T * (weights * xMat)#求出x的转置乘x的值
    if linalg.det(xTx) == 0.0:#该库函数是用于求矩阵的行列式，若行列式为0则矩阵没有逆，对w的求导公式就不成立
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

#对测试集使用局部加权线性回归进行测试，计算预测值
def lwlrTest(testArr,xArr,yArr,k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

#用于分析误差大小
def rssError(yArr,yHatArr):#输入参数为实际的y值和预测的y值
    return ((yArr - yHatArr)**2).sum()

#岭回归，当数据的特征比样本的数量还要多的时候,在xTx上加一个landa*I，I为单位矩阵使得矩阵能够求逆
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1]) * lam #获取数据的特征个数作为单位矩阵的纬度
    if linalg.det(denom) == 0.0:#该库函数是用于求矩阵的行列式，若行列式为0则矩阵没有逆，对w的求导公式就不成立
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)#此时的公式变为w = 1/(xTx + landa*I) * xT * y
    return ws

#对数据进行标准化处理，并计算在不同lamda下经过岭回归后的权值
def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)#求平均值,第二个参数为0表示一列求平均值
    xMean = mean(xMat,0)
    xVar = var(xMat,0)#求方差，第二个参数为0表示一列求方差
    #进行标准化
    yMat = yMat - yMean
    xMat = (xMat - xMean)/xVar#标准化，满足零均值和单位方差
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))#初始化一个矩阵用于存储30个不同的lamda值对应训练出来的权值
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

#前向逐步回归算法,每一步都对某个权值增加或减少一个很小的值使误差减小
def stageWise(xArr,yArr,eps=0.01,numIt = 100):#eps表示每次迭代调整权值的步长，numIt表示迭代的次数
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))#用于记录每一次迭代的w的变化
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):#每次迭代只修改一个权值
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()#wsTest用于存储每次更新的权值，由于需要尝试
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = rssError(yMat,yTest)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat




# xArr,yArr = loadDataSet('ex0.txt')
# #ws = standRegres(xArr,yArr)
# #drawPic(xArr,yArr,ws)
# print lwlr(xArr[0],xArr,yArr,1.0)
# yHat = lwlrTest(xArr,xArr,yArr,0.01)
# xMat = mat(xArr)
# yMat = mat(yArr)
# srtInd = xMat[:,1].argsort(0)
# xSort = xMat[srtInd][:,0,:]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:,1],yMat.T[:,0])#画点
# ysort = yHat[srtInd]
# ax.plot(xSort[:,1],ysort,c='r')#画线
# plt.show()
abX,abY = loadDataSet('abalone.txt')
# yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
# yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
# yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
# print rssError(abY[0:99],yHat01)#计算不同分类器对训练集预测的误差
# print rssError(abY[0:99],yHat1)
# print rssError(abY[0:99],yHat10)
# yHat01Test =lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
# yHat1Test =lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
# yHat10Test =lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
# print rssError(abY[100:199],yHat01Test)#计算不同分类器对测试集预测的误差
# print rssError(abY[100:199],yHat1Test)
# print rssError(abY[100:199],yHat10Test)
# #使用简单线性回归来训练模型
# ws = standRegres(abX[0:99],abY[0:99])
# yRegres = mat(abX[100:199]) * ws
# print rssError(abY[100:199],yRegres.T.A)
ridgeWeights = ridgeTest(abX,abY)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()














































































