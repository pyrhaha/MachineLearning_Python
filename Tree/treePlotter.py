#coding:utf-8
#使用文本注解绘制树节点
import matplotlib.pyplot as plt
decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle = "<-")
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,\
                            textcoords='axes fraction',va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()

    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode(u"决策节点",(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(u"叶节点",(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

def getNumLeafs(myTree):#获取叶节点的数目
    numLeafs = 0
    firstStr = myTree.keys()[0]#获得根节点,而且mytree只有根节点这一个key
    secondDict = myTree[firstStr]#得到根节点的值，即它的子分支的字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':#若下一个元素为字典，则表示不是叶节点
            numLeafs += getNumLeafs(secondDict[key])#使用递归搜索分支节点
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(myTree):#获取树的层数
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:#这个判断表示从一个节点出发，各分支的层数，maxDepth表示各分支中最大的层数，thisDepth对于每一个循环表示一个分支的层数
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):#用于测试存储的树的信息
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}]
    return listOfTrees[i]



#myTree = retrieveTree(0)
#print getNumLeafs(myTree)
#print getTreeDepth(myTree)
