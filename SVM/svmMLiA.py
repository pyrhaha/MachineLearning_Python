#encoding:utf-8
#支持向量机算法，通过拉格朗日乘子对1/2*(w)^2求出最小值
import random
from numpy import *
import matplotlib.pyplot as plt
from os import listdir
def loadDataSet(filename):#加载数据，将训练集加入dataMat集合，将标签加入labelMat集合
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#简化版SMO算法

#以下两个函数为SMO算法的辅助函数
#由于该算法需要调整两个a,采用的方法是一个a采用遍历所有a，另一个a采用随机选取，但不等于前一个a
def selectJrand(i,m):#该函数传入的参数为第一个a的序号，以及一共具有的a的个数，返回随机生成的一个不等于第一个a的序号
    j = i
    while (j == i):
        j = int(random.uniform(0,m))#随机生成一个从0到m之间的一个序号
    return j
#该函数用于限制alpha的值，防止其大于H或小于L
def clipAlpha(aj,H,L):#传入的参数为a的值aj，以及上边界和下边界
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj



#该函数实现了简化版的SMO算法
def  smoSimple(dataMatIn,classLabels,C,toler,maxIter):#需要输入的参数为数据集、类别标签、常数C、容错率、最大的迭代次数
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()#将标签矩阵转置
    b = 0
    m,n = shape(dataMatrix)#行数表示有多少个训练集，列数表示有多少特征
    alphas = mat(zeros((m,1)))#alpha初始化为全0
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0#记录alpha是否已经进行优化
        for i in range(m):#第一个alpha的值采用遍历的方式获得
            fXi = float( multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T) ) + b
            #计算f(x) = wx+b的值，其中w的值为w = sum( alpha(i)*y(i)*x(i) )multiply的功能是将两个矩阵的每一个对应位置上的数相乘
            Ei = fXi - float(labelMat[i])
            #以下的一些公式证明过程在统计学习方法这本书中
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C))  or  (( labelMat[i] * Ei > toler)) and (alphas[i] > 0):
                #当f(x) < 1时，该点在分界线之间，这时alpha<C，应该让alpha变大
                #当f(x) > 1时，该点在分界线外，这时alpha>0,应该让alpha减小
                j = selectJrand(i,m)#第二个alpha随机获取
                fXj = float( multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])#计算误差
                alphaIold = alphas[i].copy()#将旧的alpha值做备份
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):#i和j的符号不同，其alpha的范围也不同
                    L = max(0 , alphas[j] - alphas[i])
                    H = min(C , C + alphas[j] - alphas[i])
                else:
                    L = max(0 , alphas[j] + alphas[i] - C)
                    H = max(C , alphas[j] + alphas[i])
                if L == H:
                    print 'L=H'
                    continue
                eta = dataMatrix[i,:]*dataMatrix[i,:].T + dataMatrix[j,:] * dataMatrix[j,:].T - 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T
                #eta为最优修改量，在使用了核函数时的计算公式为K(11)+K(22)-2K(12),这个例子没有使用核函数所以就等于x1^2 + x2^2 - 2*x1*x2
                if eta <= 0:
                    print "eta <= 0 "
                    continue
                alphas[j] += labelMat[j] * (Ei - Ej)/eta#对alpha(2)进行更新
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001):#alpha的修改量太小
                    iter+=1
                    print 'j not moving enough'
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])#对alpha(1)进行更新
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:]*dataMatrix[i,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:]*dataMatrix[j,:].T
                #计算b的公式为
                # b1 = b(old) - Ei - y1*K(11)*(alpha(new1) - alpha(old1)) - y2*K(21)(alpha(new2) - alpha(old2))
                # b2 = b(old) - Ej - y1*K(12)*(alpha(new1) - alpha(old1)) - y2*K(22)(alpha(new2) - alpha(old2))
                if (alphas[i] > 0) and (alphas[i] < C):
                    b = b1
                elif(alphas[j] > 0) and (alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print 'iter:%d  i:%d, pairs changed %d' % (iter,i,alphaPairsChanged)
        if alphaPairsChanged == 0 :#若本次迭代alpha没有被优化，则将iter+1，直到iter达到maxIter退出循环，否则将iter置0，继续循环
            iter += 1
        else:
            iter = 0
        print alphas
        #只有在所有数据集上遍历maxIter次，且不再发生任何alpha值修改后，程序才会退出while循环
        print 'iteration number: %d' %iter
    #w =  sum(multiply(multiply(alphas,labelMat),dataMatrix))
    w = calcWs(alphas,dataMatrix,labelMat)
    return b,alphas,w







def plot(w,b,dataArr,labelMat):#绘制图像
    dataArr = array(dataArr)
    b = array(b)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,0])
            ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0])
            ycord2.append(dataArr[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    #为了拟合分隔线，0=w0*x0+w1*x1+w2*x2,其中w0为常数项，所以x0为1，x1为x，x2为y,拟合出x1和x2之间的关系
    x = arange(-5.0,15.0,1)
    y = (-w[0]*x-b)/w[1]
    ax.plot(x,y[0])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


#以下是完整版的SMO算法，其主要改进了alpha选择的部分
#optStruct这个类用于创建一个数据结构来存储重要的数据,只含有__init__这一个函数
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):#初始化各个参数
        self.X = dataMatIn #训练数据集
        self.labelMat = classLabels #标签集
        self.C = C #常数C，作为惩罚参数，用于调和间隔尽量大以及误分类点的数量之间的关系
        self.tol = toler #容错率
        self.m = shape(dataMatIn)[0] #训练集的大小
        self.alphas = mat(zeros((self.m,1))) #需要训练的参数alpha
        self.b = 0 #需要求的分界线的偏移量b
        self.eCache = mat(zeros((self.m,2)))#用于对误差E进行缓存，第一列存储是否有效的标志位，第二列存储实际的E
        self.k = mat(zeros((self.m,self.m)))#用于储存核函数的矩阵,在初始化的时候就计算了k的值，训练的时候直接取
        for i in range(self.m):
            self.k[:,i] = kernelTrans(self.X,self.X[i],kTup)

def calcEk(oS,k):#计算误差E，传入的参数中oS是该类的一个实例,k表示训练数据是训练集的哪一个数据
    fXk =float(multiply(oS.alphas,oS.labelMat).T * (oS.k[:,k])) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):#选择第二个alpha的启发式算法
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    vaildEcacheList = nonzero(oS.eCache[:,0].A)[0]#获取ecache中不为零的索引列表,nonzero函数接收参数为一个列表，返回值是一个元组，该元组的两个元素是两个列表，表示第一个列表存放所有非零元素的行数，第二个列表存放列数
    if(len(vaildEcacheList)) > 1:
        for k in vaildEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:#如果这是第一次循环，所以vaildEcacheList里面没有其他东西，就随机选择一个
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
    return j, Ej

def updateEk(oS,k):#用于在alpha优化完以后更新缓存Ek
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]



def innerL(i,oS):#该函数用于优化alpha，包括两个alpha的选择，思路和简化版很相似只是将selectJrand()函数换成了selectJ()函数,最后更新误差缓存
    Ei = calcEk(oS,i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print "L == H"
            return 0
        #eta = oS.X[i,:]*oS.X[i,:].T + oS.X[j,:]*oS.X[j,:].T -  2.0 * oS.X[i,:] * oS.X[j,:].T#最原始的式子
        eta = oS.k[i,i] + oS.k[j,j] -  2.0 * oS.k[i,j] #使用核函数后的式子
        if eta <= 0:
            print "eta <= 0"
            return 0
        oS.alphas[j] += oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough"
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS,i)
        #b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i,:]*oS.X[j,:].T#这是原始的式子
        #b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j,:]*oS.X[j,:].T#这是原始的式子
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.k[i,i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.k[i,j]#使用核函数
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.k[i,j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.k[j,j]#使用核函数
        #计算b的公式为：
        # b1 = b(old) - Ei - y1*K(11)*(alpha(new1) - alpha(old1)) - y2*K(21)(alpha(new2) - alpha(old2))
        # b2 = b(old) - Ej - y1*K(12)*(alpha(new1) - alpha(old1)) - y2*K(22)(alpha(new2) - alpha(old2))
        if (oS.alphas[i] > 0) and (oS.alphas[i] < oS.C):
            oS.b = b1
        elif(oS.alphas[j] > 0) and (oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin',0)):#完整的SMO算法中的外部循环
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler,kTup)
    iter = 0
    entireSet = True#用于标记是否是遍历全部数据集更新alpha
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:#完整版的SMO算法获取第一个alpha值采用（遍历所有数据集）和（遍历不等于0或C的alpha值，即非边界值）交替的方法
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)#表示被修改的alpha数
                print "fullset, iter:%d  i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0] #此处nonzero的用法是返回alpha列表中在0~C之间的值的索引元组
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter:%d  i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:#如果选择那些非边界点更新的方法，当没有可以更新的alpha时，下一次迭代改为循环整个训练集
            entireSet = True
        print "iteration number: %d" % iter
    w = calcWs(oS.alphas,oS.X,oS.labelMat)
    return oS.b, oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels)
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i].transpose(),X[i,:].T)
    return w


def kernelTrans(X,A,kTup):#将SVM扩展到使用核函数
    m,n = shape(X)
    k = mat(zeros((m,1)))
    if kTup[0] == "lin":
        k = X * A.T #此时为最简单的内积
    elif kTup[0] == "rbf":
        for j in range(m):#使用径向基函数高斯版作为核函数，其具体公式为 k(x,y) = exp( -(x-y)^2 / 2*theta^2) ,其中theta为用户自定义的到达率
            deltaRow = X[j,:] - A
            k[j] = deltaRow * deltaRow.T
        k = exp( k / (-1 * kTup[1]**2) )
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return k

def testRbf(k1 = 1.3):#测试该分类器
    #使用训练集测试
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]#找到alpha > 0的点，因为alpha=0的点在计算w的时候无效，所以只需提取>0的点
    svs = dataMat[svInd]#获得那些使alpha不为0的数据
    labelSV = labelMat[svInd]
    m,n = shape(dataMat)
    errorcount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs,dataMat[i],('rbf',k1))#计算K(x,y)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b #需要计算的分类值=w*K(x,y)+b
        if sign(predict) != sign(labelArr[i]):#sign函数使x>0则输出1，x>0则输出-1
            errorcount+=1
    print "the training error rate is: %f" % (float(errorcount)/m)
    #使用测试集测试，和用训练集测试的代码就数据集不同，其他都一样
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]#找到alpha > 0的点，因为alpha=0的点在计算w的时候无效，所以只需提取>0的点
    svs = dataMat[svInd]#获得那些使alpha不为0的数据
    labelSV = labelMat[svInd]
    m,n = shape(dataMat)
    errorcount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs,dataMat[i],('rbf',k1))#计算K(x,y)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b #需要计算的分类值=w*K(x,y)+b
        if sign(predict) != sign(labelArr[i]):#sign函数使x>0则输出1，x>0则输出-1
            errorcount+=1
    print "the test error rate is: %f" % (float(errorcount)/m)

#采用支持向量机进行手写字识别，由于支持向量机是二类分类器,我们的训练集只有0和9

def img2vector(filename):#将32*32的图像转换为[1,1024]的向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):#循环读出文件的前32行
        lineStr = fr.readline()
        for j in range(32):#每行的头32个字符
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirname):#该函数返回训练集和标签
    hwLabel = []
    trainingFileList = listdir(dirname)
    m = len(trainingFileList)#获取训练集的数量
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNamestr = trainingFileList[i]
        fileStr = fileNamestr.split(".")[0]
        className = fileStr.split("_")[0]
        if className == '9':#我们的训练集只有0和9
            hwLabel.append(-1)
        else:
            hwLabel.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirname,fileNamestr))
    return trainingMat,hwLabel

def testDigits(kTup = ('rbf',10)):
    #使用训练集测试
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]#找到alpha > 0的点，因为alpha=0的点在计算w的时候无效，所以只需提取>0的点
    svs = dataMat[svInd]#获得那些使alpha不为0的数据
    labelSV = labelMat[svInd]
    m,n = shape(dataMat)
    errorcount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs,dataMat[i],kTup)#计算K(x,y)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b #需要计算的分类值=w*K(x,y)+b
        if sign(predict) != sign(labelArr[i]):#sign函数使x>0则输出1，x>0则输出-1
            errorcount+=1
    print "the training error rate is: %f" % (float(errorcount)/m)
    #使用测试集测试，和用训练集测试的代码就数据集不同，其他都一样
    dataArr,labelArr = loadImages('testDigits')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]#找到alpha > 0的点，因为alpha=0的点在计算w的时候无效，所以只需提取>0的点
    svs = dataMat[svInd]#获得那些使alpha不为0的数据
    labelSV = labelMat[svInd]
    m,n = shape(dataMat)
    errorcount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs,dataMat[i],kTup)#计算K(x,y)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b #需要计算的分类值=w*K(x,y)+b
        if sign(predict) != sign(labelArr[i]):#sign函数使x>0则输出1，x>0则输出-1
            errorcount+=1
    print "the test error rate is: %f" % (float(errorcount)/m)

#dataArr,labelArr = loadDataSet('testSet.txt')
#b,alphas,w = smoP(dataArr,labelArr,0.6,0.001,15)
#print w,b
#print alphas[alphas>0]
#plot(w,b,dataArr,labelArr)
testDigits(('rbf',20))