#encoding:utf-8
#使用Adaboost提升分类器的准确率，通过为不同的分类器设置权重，以及不断改变每个训练样本的权值来训练模型，将分类错误的样本的权值提高
from numpy import *
from boost import *
import matplotlib.pyplot as plt
#加载数据
def loadSimpData():
    dataMat = mat([
                        [1.,2.1],
                        [2.,1.1],
                        [1.3,1.],
                        [1.,1.],
                        [2.,1.]
                        ])
    classMat = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classMat

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



#adaboost模型的训练
def adaBoostTrainDS(dataArr,classLabels,numIter = 40):
    weakClassArr = []#一个存放多个单层决策树的列表
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)#权值向量，初始化为相同的值
    aggClassEst = mat(zeros((m,1)))#存储多个模型组合最后求出的值，最后经过sign函数即为用该分类器分类的结果
    for i in range(numIter):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#经过单层决策树分类的结果
        print "The error of this classifier is:",error
        print "D:",D.T
        alpha = float(0.5*log10((1.0-error)/max(error,1e-16)))#根据error来更新alpha,公式为alpha=（1/2）*ln( (1-error)/error ),其中max(error,1e-16)是为了防止当错误率为0时出现被0除的错误
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)#添加一个新的决策树到分类器中
        print "classEst:",classEst
        #更新权值向量与预测的标签和实际标签的关系有关，如果分类正确，则该数据集的权值赋值为D = D * e^(-alpha) / D(sum),若错误分类，则赋值为D = D * e^(alpha) / D(sum)，即和预测标签和实际标签是否同号有关
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D / D.sum()
        aggClassEst += alpha*classEst#将各分类器的结果与alpha做积求和得到分类结果
        print "aggClassEst:",aggClassEst
        #计算错误率：若通过各分类器结果求和得到的值再经过sign函数得到的1或-1值与标签相同则矩阵的该分量为0，否则为1，最后求和得到的就是错误分类的数量
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T , ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error:",errorRate,"\n"
        if errorRate == 0.0:
            break
    return weakClassArr,aggClassEst

#使用adaBoost分类器对数据进行分类
def adaClassify(datToClass,classifierArr):#传入的参数为需要分类的数据和一系列分类器的列表
    dataMat = mat(datToClass)
    m = shape(dataMat)[0]
    aggClassEst = zeros((m,1))
    for i in range(len(classifierArr)):#遍历所有分类器
        classEst = stumpClassify(dataMat,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)

#在马疝病数据集上使用adaboost分类器
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

#绘制ROC曲线，计算AUC的值
def plotROC(predStreangths, classLabels):#第一个参数为经过AdaBoost或SVM分类器计算的函数值（即在分类器中需要会经过sign()函数进行二类分类），第二个参数为数据实际的标签
    cur = (1.0,1.0)#用于记录光标的位置，初始化位置为（1,1），在坐标轴的右上角，即将所有的实例都分类成正例
    ySum = 0.0#记录AUC的值，即ROC曲线下的面积
    numPosClass = sum(array(classLabels) == 1.0)#使用数组过滤来计算正例的个数
    yStep = 1/float(numPosClass)#坐标纵轴的步长为正例个数的倒数，即当所有的正例都被找到则y=0
    xStep = 1/float(len(classLabels) - numPosClass)#坐标纵轴的步长为反例个数的倒数
    sortedIndicies = predStreangths.argsort()#对分类器计算值从小到大进行排序，获取排序后的索引值,从小到大排序使得尽可能将标签为1的项放在后面，这样如果是好的分类器会先迅速的减小x轴的值，然后碰到一个标签为1的减小y轴的值，画的图越接近左上角越好
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')#进行绘制，绘制一条斜线
        cur = (cur[0]-delX,cur[1]-delY)#改变此时的光标
    ax.plot([0,1],[0,1],'b--')#该线为随机猜测的ROC
    plt.xlabel('False Postitive Rate')
    plt.ylabel('True Postitive Rate')
    plt.title('ROC curve for adaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print "The AUC is ",ySum*xStep




dataMat,labelMat = loadDataSet('horseColicTraining2.txt')
classifierArr,aggClassEst = adaBoostTrainDS(dataMat,labelMat,9)
#adaClassify([0,0],classifierArr)
#testMat,testLabel = loadDataSet('horseColicTest2.txt')
#prediction = adaClassify(testMat,classifierArr)
#m = shape(testMat)[0]
#errArr = mat(ones((m,1)))
#err =  errArr[prediction != mat(testLabel).T].sum()
#print err/m
plotROC(aggClassEst.T,labelMat)