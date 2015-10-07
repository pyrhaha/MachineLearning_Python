#coding:utf-8
from math import log
import operator
import pickle
import treePlotterAnswer

def calcShannonEnt(dataSet):#计算数据集的熵
    numEntries = len(dataSet)
    labelCounts = {}#用于存储各个标签对应的出现次数，即选择该分类的概率
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)#熵为所有分类的信息期望值之和，设选择该分类的概率为x，则信息期望值= -x*log2(x),由于对数的底数为2，所以熵的单位为bit
    return shannonEnt

def createDataSet():#设置数据集
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet, axis, value):#该函数用于按照给定特征划分数据集，三个参数分别为待划分的的数据集、划分数据集的特征的索引、当该特征的取值为value时被选择
    retDataSet = []#由于python函数传列表参数，传的是列表的引用，所以为了防止函数里对列表的修改会影响原列表，创建了一个新的列表
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFearVec = featVec[:axis]
            reducedFearVec.extend(featVec[axis+1:])#这两条语句将指定特征为value的数据集选择出来，并去掉了该特征值
            retDataSet.append(reducedFearVec)
            #列表的append和extend函数的区别为：如a=[1,2,3],b=[4,5,6],a.append(b)结果为[1,2,3,[4,5,6]]，a.extend(b)结果为[1,2,3,4,5,6]

    return retDataSet

def chooseBestFeatureToSplit(dataSet):#选择最好的数据划分方式,返回选择的特征值在数据集中的索引
    numFeatures = len(dataSet[0]) - 1#该值为特征的个数，去掉最后一个为分类结果
    baseEntropy = calcShannonEnt(dataSet)#计算初始的熵
    bestInfoGain = 0.0#信息增益即熵减少量
    bestFeature = -1#初始化最大的熵减少量
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#使用列表推导将所有第i个特征值的取值取出
        uniqueVals = set(featList)#set函数将列表转为集合，消除重复的元素
        newEntropy = 0.0
        for featVal in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,featVal)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):#采用多数表决法确定叶子节点的分类,传入的标签为数据集中的所有类标签
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):#使用递归创建树,传入的参数中labels表示特征的名字，在分类中没有用到，只是为了显示树的时候更加直观，
    classList = [example[-1] for example in dataSet]#将数据集中的类别所有取值提取出来
    if classList.count(classList[0]) == len(classList):
    #如果所有的类标签都是一样，则都为classList[0]，即classList长度和classList中classList[0]的个数相同

        return classList[0]#所有类标签完全相同时，直接返回该类别
    if len(dataSet[0]) == 1:#当数据集只有一列，即只有一个特征值，但又不是同样的分类标签，则用多数表决法确定叶子节点的分类
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]#选择的标签
    myTree = {bestFeatLabel:{}}#使用字典来表示树
    del(labels[bestFeat])#将该特征从类标签中除去
    featValues = [example[bestFeat] for example in dataSet]#将数据集中该特征值的取值取出
    uniqueVals = set(featValues)#去除重复
    for value in uniqueVals:
        subLabels = labels[:]#由于列表是引用传递，避免对列表的修改，拷贝一个新的
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):#使用决策树的分类函数，参数为构建的决策树，各特征值的标签，需要分类的数据集
    firstStr = inputTree.keys()[0]#获得根节点,而且mytree只有根节点这一个key
    secondDict = inputTree[firstStr]#得到根节点的值，即它的子分支的字典
    featIndex = featLabels.index(firstStr)#将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':#该节点不是叶节点
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):#将决策树序列化后存入文件
    fw = open(filename,"w")
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):#从文件中将决策树提取出来
    fr = open(filename)
    return pickle.load(fr)

def drawGlassTree():#配眼镜问题画决策树的实例
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabel = ['age','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,lensesLabel)
    treePlotterAnswer.createPlot(lensesTree)




#myDat,labels = createDataSet()
#shannonEnt = calcShannonEnt(myDat)
#print shannonEnt
#copyLabels = labels[:]
#myTree = createTree(myDat,copyLabels)
#copyLabels = labels[:]
#print classify(myTree,copyLabels,[1,1])
drawGlassTree()