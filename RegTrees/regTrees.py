#coding:utf-8
#构建树回归模型，使用CART算法，既可以用于回归也可以用于分类，有一些数据进行全局建模无法实现时，可以使用回归树将数据分割成几个部分分别进行建模
from numpy import *
from Tkinter import *
##############################################################################
#回归树的切分函数
def regLeaf(dataSet):
    return mean(dataSet[:,-1])#计算叶节点为目标变量的均值

#回归树的误差函数
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]#误差估计函数，在给定数据上计算目标变量的平方误差，也就是方差，var能够计算均方误差，乘上数据集的数量就是总方差

#搜索最合适用于切分的特征
def chooseBestSplit(dataSet,leafType=regLeaf,errType = regErr,ops=(1,4)):#最后一个参数是一个元组，分别表示容许的误差下降值和切分的最小样本数，这是用于预剪枝的，使得分割数据的过程提前结束
    tolS = ops[0]#容许的误差下降值
    tolN = ops[1]#切分的最小样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:#对于一些离散型的数据，当剩下的数据集的目标变量取值只有一个时就不需要再切分了
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)#计算原先的误差值
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):#遍历所有的特征
        for splitVal in set(dataSet[:,featIndex]):#遍历该特征的所有取值
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):#若分割后的子集小于设定的切分最小样本数，则跳过
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:#若误差下降值变化小于设定的阈值，则不分割
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):#加入这句判断是为了防止遍历所有的特征也找不到可以分割的特征，这时需要直接返回
        return None,leafType(dataSet)
    return bestIndex,bestValue

#该函数用于判断是否是叶节点
def isTree(obj):
    return (type(obj).__name__ == 'dict')

#将两个分支合并，即求两个叶节点的平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])#使用递归的方式获取右边的叶节点
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2#进行塌陷处理，即对子树求平均值

#该函数使用测试数据进行后剪枝
def prune(tree,testData):#参数为创建的树和测试数据
    if shape(testData)[0] == 0:#如果该子树没有测试集，则合并子树
        return getMean(tree)
    lSet,rSet = binSplitDataSet(testData,tree['splitInd'],tree['splitVal'])
    #if isTree(tree['left']) or isTree(tree['right']):#若左右子树中至少存在一个树

    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        #lSet,rSet = binSplitDataSet(testData,tree['splitInd'],tree['splitVal'])
        errorNoMerge = sum( power( lSet[:,-1] - tree['left'],2))+ sum(power(rSet[:-1] - tree['right'],2))#计算未合并之前的误差采用目标变量和叶节点的差的平方
        treeMean = getMean(tree)
        errorMerge = sum(power(testData[:,-1] - treeMean,2))#这是合并后的误差
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


##############################################################################
#模型树的叶节点生成函数
#使用训练集计算线性拟合函数的权值ws
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]#由于dataSet的最后一列是目标变量，所以只需要前n-1列，X的第一列全为1用于计算常数项
    Y = dataSet[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:#该库函数是用于求矩阵的行列式，若行列式为0则矩阵没有逆，对w的求导公式就不成立
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

#为叶节点创建模型，即线性方程
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

#计算模型的误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(yHat-Y,2))
##############################################################################

#以下代码是构建回归树和模型树可以共用的代码，在回归树中每个叶节点是一个值，在模型树中每一个叶节点是一个线性方程

#加载数据
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)#将curLine中的元素映射到float中，该函数的功能相当于将curLine中的每一个元素使用float函数进行处理
        dataMat.append(fltLine)
    return dataMat

#分割数据，参数分别为数据集，选择的特征，分隔数据的特征的值
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]#使用数组过滤的方法进行分隔,由于这里是对多维矩阵进行过滤，所以使用nonzero函数
    mat1 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    return mat0,mat1

#该函数用于生成决策树
def createTree(dataSet,leafType = regLeaf,errType = regErr, ops=(1,4)):#第一个参数表示数据集，第二个参数表示要生成的树的类型，因为回归树和模型树的一个区别在于，即建立叶节点的函数，第三个参数表示误差计算的函数，第四个参数是一个元组，其中包含构建树需要的其他参数
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)#调用该函数返回切分使用的特征以及切分的特征值
    if feat == None:#如果该特征不需要切分，则返回特征值，该子节点为叶节点
        return val
    retTree = {}
    retTree['splitInd'] = feat#若需要切分，则子节点是一个子树，该子树的splitInd存储的是切分选择的特征的索引
    retTree['splitVal'] = val#存储分隔值
    lSet,rSet = binSplitDataSet(dataSet,feat,val)#对数据集进行分隔
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

#用树回归进行预测的一系列函数
#回归树获取预测值
def regTreeEval(model,inDat):#虽然只使用了一个参数，但是为了和模型树的格式一致，所以传入两个参数
    return float(model)

#模型树获取预测值
def modelTreeEval(model,inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

#使用递归对一个数据进行预测
def treeForeCast(tree,inData,modelEval = regTreeEval):
    if not isTree(tree):#到达叶节点，获取预测值
        return modelEval(tree,inData)
    if inData[tree['splitInd']] > tree['splitVal']:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)
    else:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)

#对测试集的所有数据进行预测
def createForeCast(tree,testData,modelEval = regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,testData[i],modelEval)
    return yHat



# testMat = mat(eye(4))
# mat0,mat1 = binSplitDataSet(testMat,1,0.5)
# print mat0
# myDat = loadDataSet('exp2.txt')
# myMat = mat(myDat)
# print createTree(myMat,modelLeaf,modelErr)
# myDataTest = loadDataSet('ex2test.txt')
# myMatTest = mat(myDataTest)
# MergeTree = prune(myTree,myMatTest)
# print MergeTree
##############################################################################
#使用相关系数来评价回归树的效果
trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
myTree = createTree(trainMat,ops=(1,20))
yHat = createForeCast(myTree,testMat[:,0])
cor = corrcoef(yHat,testMat[:,-1],rowvar=0)[0,1]
print cor
#使用相关系数来评价模型树的效果
trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
myTree = createTree(trainMat,modelLeaf,modelErr,(1,20))
yHat = createForeCast(myTree,testMat[:,0],modelTreeEval)
cor = corrcoef(yHat,testMat[:,-1],rowvar=0)[0,1]
print cor
#使用相关系数来评价简单线性回归的效果
ws,X,Y = linearSolve(trainMat)
for i in range(shape(testMat)[0]):
    yHat[i] = testMat[i,0]*ws[1,0] + ws[0,0]
print corrcoef(yHat,testMat[:,-1],rowvar=0)[0,1]