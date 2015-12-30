#coding:utf-8
#该文件是一个FP-growth算法，这是一个比Apriori高效的频繁集发现算法，它实现的原理是构建FP树，
#这样在提取出频繁项的时候不用每次都遍历整个数据集来计算支持度，只用遍历两次就可以了
#更加具体的图见机器学习P227中的图
#创建一个FP树的类
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):#定义节点包含的属性
        self.name = nameValue#节点的名字
        self.count = numOccur#节点的计数器用于记录出现的次数
        self.nodeLink = None#用于链接相似的元素项
        self.parent = parentNode#存储父节点的指针
        self.children = {}#存储子节点

    #对count变量增加给定值
    def inc(self,numOccur):
        self.count += numOccur

    #将树以文本形式显示
    def disp(self,ind=1):
        print ' '*ind,self.name, ' ',self.count#打印出节点信息，包括节点名字和出现的个数
        for child in self.children.values():
            child.disp(ind+1)#ind变量表示树的层数，使用递归遍历所有子节点

#更新头指针表
def updateHeader(preNode,targetNode):
    while preNode.nodeLink != None:#直到指向最后一个节点,将新加的节点加在后面
        preNode = preNode.nodeLink
    preNode.nodeLink = targetNode


#更新扩展FP树
def updateTree(items,inTree,headerTable,count):#传入的参数为items事务的元素项的列表，inTree为树的根节点，header头指针表，count该事务的次数
    if items[0] in inTree.children:#如果该元素项是该节点的子节点，就将count直接加在原来的值上
        inTree.children[items[0]].inc(count)
    else:#否则创建一个新的节点，添加到根节点的子节点中
        inTree.children[items[0]] = treeNode(items[0],count,inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])#由于添加了新的节点，所以在头指针表中需要能够指向该节点
    if len(items) > 1:
        updateTree(items[1:],inTree.children[items[0]],headerTable,count)
        #递归调用updateTree函数将该事务中的所有元素项加入到树中





#FP树的构建函数
def createTree(dataSet,minSup=1):#参数分别代表数据集和最小支持度阈值，数据集是一个字典，在loadSimpleData中初始化键为一个事务的字典，值为1，当构建条件FP树的时候这个值不一定为1
    headerTable = {}#头指针表，字典的值是一个元组，分别表示(出现的频数，头指针)
    for transactions in dataSet:
        for item in transactions:
            headerTable[item] = headerTable.get(item,0) + dataSet[transactions]#这里先将headerTable只存储频数，所以将频数+该事务出现的次数
    for k in headerTable.keys():
        if headerTable[k] < minSup:#如果不是频繁集就将其从头指针表中删除
            del headerTable[k]
    freqItemSet = set(headerTable)
    if len(freqItemSet) == 0:
        return None,None
    for item in headerTable:
        headerTable[item] = [headerTable[item],None]
    retTree = treeNode('Null Set',1,None)
    for transaction,count in dataSet.items():#遍历数据集中的每一个事务，一次循环只针对一个事务中的元素进行处理
        localData = {}#临时的变量，用于存储一个元素项对应的频数
        for item in transaction:#遍历事务中的所有元素
            if item in freqItemSet:#如果该元素项是频繁项集，则将localData中的该元素对应的值赋为该元素项的频数
                localData[item] = headerTable[item][0]
        if len(localData) > 0:
            orderedItems = [v[0] for v in sorted(localData.items(),key= lambda p:p[1],reverse=True)]
            #对一个事务中的所有元素项根据频数进行从大到小的排序，localData.items()获取(键，值)的元组，所以根据p[1]进行排序，orderedItems中获取的值为v[0]
            updateTree(orderedItems,retTree,headerTable,count)#调用updateTree去更新扩展FP树，其中count表示该事务出现的次数
    return retTree,headerTable

#加载一个简单的数据集
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

#由于数据集是一个列表，不满足算法要求的数据集是一个字典所以对数据进行格式化
def createInitSet(dataSet):
    retDict = {}
    for transaction in dataSet:
        retDict[frozenset(transaction)] = 1
    return retDict

#获得一个节点的所有前缀节点
def ascendTree(leafNode,prefixPath):#传入的参数分别表示开始回溯的节点和需要添加的前缀的列表
    while leafNode.parent != None:#因为根节点是一个‘Null Set’节点，所以不应该将其加入列表
        prefixPath.append(leafNode.name)
        leafNode = leafNode.parent

#为给定元素项生成一个条件模式基，即找到相同元素项的前缀节点
def findPrefixPath(basePath,treeNode):#传入的参数为给定的元素项(该项中可以有很多元素)和该元素项对应在树中的节点
    condPaths = {}#条件模式基
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) > 1:#由于该列表中包含元素本身，所以当长度大于1时，存在前缀
            condPaths[frozenset(prefixPath[1:])] = treeNode.count#条件模式基的键为前缀，值为频数
        treeNode = treeNode.nodeLink
    return condPaths

#递归查找频繁项集的函数，和Apriori的思路差不多，但是它不生成候选集，而是通过从树的一个节点开始往上遍历来增加一个元素项的大小
#该函数的思路有点难理解，它首先通过传入preSet为空，headerTable中都是频繁集，所以此时遍历的每一个元素都是频繁集，
#然后查找所有前缀，并构造条件树和头指针表，在这个过程中就过滤掉了不频繁项，所以头指针表中的元素就是频繁集，
#递归时将头指针的所有元素遍历，因为已经确定是频繁集，所以加入freqItemList中，再继续搜索前缀，直到没有前缀
def mineTree(inTree,headerTable,minSup,preSet,freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda p: p[1])]#对头指针表中的每一个元素项进行从小到大排序(因为构建树的时候是从大到小的),都是频繁集
    for basePath in bigL:#遍历传入频繁集的每个元素,在每次循环中作为基础元素
        newFreqSet = preSet.copy()
        newFreqSet.add(basePath)
        freqItemList.append(newFreqSet)
        prePathList = findPrefixPath(basePath,headerTable[basePath][1])#得到该元素节点的所有前缀，是一个列表，每一个元素是一个前缀的列表
        myCondTree,myHeadTable = createTree(prePathList,minSup)
        if myHeadTable != None:
            print 'conditional tree for: ',newFreqSet
            myCondTree.disp()
            mineTree(myCondTree,myHeadTable,minSup,newFreqSet,freqItemList)



# rootNode = treeNode('pyramid',9,None)
# rootNode.children['eye'] = treeNode('eye',13,None)
# rootNode.disp()
# simpData = loadSimpDat()
# initSet = createInitSet(simpData)
# myFPtree,myHeaderTable = createTree(initSet,3)
# #myFPtree.disp()
# #print findPrefixPath('r',myHeaderTable['r'][1])
# freqItems = []
# mineTree(myFPtree,myHeaderTable,3,set([]),freqItems)
parseData = [line.split() for line in open('kosarak.dat').readlines() ]
initData = createInitSet(parseData)
myTree,myHeadTable = createTree(initData,100000)
myFreqList = []
mineTree(myTree,myHeadTable,100000,set([]),myFreqList)
print myFreqList