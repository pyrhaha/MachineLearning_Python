#encoding:utf-8
#该算法是通过使用Apriori算法进行关联分析
#首先是使用Apriori算法来发现频繁集，该算法的目的是避免在搜索频繁项的时候需要计算的数据太多，
#该算法的原理是若某个集合是非频繁集，则其超集(就是指包含该集合的集合)也是非频繁集，
#所以从集合的元素数量由小到大搜索时，如果碰到某集合是非频繁集，则就不需要再搜索包含该集合的其他组合了，减少了计算量

##################################################################################################
#Apriori获取频繁项集
#Apriori的一些支持函数
#建立简单的数据集
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

#构建集合C1，C1是大小为1的项集(即每一个集合中只有一项)的所有候选项集的集合，是搜索所有数据集的第一步
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:#数据集中的每一项为一个事务
        for item in transaction:#每一个事务中包含一系列的元素
            if not [item] in C1:#如果C1中没有该项集，则加入
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)#因为后面每一个项集需要作为字典的键值，所以将其变成forzenset类型，即冰冻的集合，不可改变

#搜索数据集计算支持度，并过滤掉小于设定的最小支持度的项集
def scanD(D,Ck,minSupport):#传入的三个参数分别为数据集D，候选项集列表Ck，设定的最小支持度
    SSCnt = {}#一个字典，用于存储计算出来的各个项集在事务中出现的个数
    for transaction in D:#遍历数据集中的每一个事务
        for candidate in Ck:#遍历每一个候选项集
            if candidate.issubset(transaction):#如果候选项集是该事务的子集
                if not SSCnt.has_key(candidate):#如果SSCnt中没有该项则为该项赋值为1
                    SSCnt[candidate] = 1
                else:#如果SSCnt中有了，则出现的个数+1
                    SSCnt[candidate]+=1
    numTransaction = float(len(D))#获取事务的总数
    NewCk = []#用于存储满足最小支持度的项集
    supportData = {}#用于存储过滤后剩下的项集的支持度
    for item in SSCnt:
        support = SSCnt[item]/numTransaction#计算支持度
        if support >= minSupport:
            NewCk.insert(0,item)#使用insert函数将项集加入到NewCk前面
            supportData[item] = support
    return NewCk,supportData#该函数返回满足支持度限制的项集集合和不同项集对应的支持度


#使用apriori算法
#定义每一次迭代如何合并生成新的项集，思路是当两个项集的前k-2项相同时，将两个集合合并,每次合并多一个元素
def aprioriGen(Lk,k):#传入的参数含义为频繁项集列表Lk,项集的元素个数
    NewLk = []
    lenLk = len(Lk)
    for i in range(lenLk):#遍历所有的频繁项集
        for j in range(i+1,lenLk):#让选择的频繁项集1与后面的其他频繁项集2组合
            L1 = list(Lk[i])[:k-2]#提取频繁项集1的前k-2项
            L2 = list(Lk[j])[:k-2]#提取频繁项集2的前k-2项
            L1.sort()
            L2.sort()
            if L1 == L2:#若两个项集的前k-2元素相等，则只有最后一个元素是不同的，那么两个项集求并集则可以得到一个含有k+1个元素的项集
                NewLk.append(Lk[i] | Lk[j])
    return NewLk

#Apriori算法的主函数
def apriori(dataSet,minSupport = 0.5):#传入的参数为数据集和设定的最小支持度
    C1 = createC1(dataSet)
    D = map(set,dataSet)
    L1,supportData = scanD(D,C1,minSupport)
    frequentList = [L1]#用于存储生成的所有频繁项集，每一个元素也是一个列表，每个元素存的是一系列项数相同的频繁集
    k = 2
    while (len((frequentList[k-2])) > 0):#遍历每一个数量的项数集，直到某一个数量的项数集为空时结束循环
        cadidateSet = aprioriGen(frequentList[k-2],k)#合并集合获取候选集
        freqSet,support = scanD(D,cadidateSet,minSupport)#获取频繁集
        supportData.update(support)#对字典进行更新，添加新的满足支持度条件的项集的支持度信息
        frequentList.append(freqSet)
        k+=1
    return frequentList,supportData

################################################################################################

#获取关联规则

#对规则进行评估，过滤掉小于可信度阈值的关联规则
def calcConf(freqSet,H,supportData,brl,minConf):#传入的参数为频繁集列表，将该频繁集分成一个个元素的列表，频繁集的支持度信息，主函数中最后需要返回的关联规则，可信度阈值
    prunedH = []#用于存储后件，作为返回值返回，接着会通过aprioriGen函数合成新的后件列表
    for backConseq in H:
        conf = supportData[freqSet] / supportData[freqSet-backConseq]#可信度的计算公式为P+H/P,其中P为前件，H为后件,关系为P->H
        if conf >= minConf:
            print freqSet-backConseq,'--->',backConseq,'conf:',conf
            brl.append((freqSet-backConseq,backConseq,conf))#添加关联规则(前件，后件，可信度)，加入到主函数的brl中
            prunedH.append(backConseq)#将后件加入列表
    return prunedH

#对于一个频繁集，通过递归的方式合并增加后件元素的个数不断创建新的关联规则
def rulesFromBackConseq(freqSet,H,supportData,brl,minConf=0.7):#
    m = len(H[0])#获取现在后件中的元素个数
    if (len(freqSet) > (m+1)):#如果m+1,即合并后的后件个数小于频繁集的元素个数
        NewBackConseq = aprioriGen(H,m+1)#调用aprioriGen函数合并后件，得到新的后件候选集
        NewBackConseq = calcConf(freqSet,NewBackConseq,supportData,brl,minConf)#对后件的候选集进行评估，筛选出满足条件的后件列表
        if (len(NewBackConseq) > 1):#如果此时后件集中还有两个以上，说明还可以通过合并形成新的后件
            rulesFromBackConseq(freqSet,NewBackConseq,supportData,brl,minConf)#递归形成新的后件



#生成关联关系列表的主函数
def generateRules(L,supportData,minConf = 0.7):#传入的三个参数分别为频繁项集列表，频繁项集的支持度信息的列表，设置的最小可信度阈值
    bigRuleList = []#用于存储生成的所有关联规则
    for i in range(1,len(L)):#遍历所有频繁项集L，一个L的例子为[ [ [0],[1],[2] ],[ [0,1],[1,2],[0,2] ] ]
        #所以每一个L[i]为，比如[ [0],[1],[2] ]
        #因为只有一个元素的频繁集无法找关联规则，所以从1开始
        for freqSet in L[i]:#遍历每一个频繁集，比如[0]或[0,1]
            H1 = [ frozenset([item]) for item in freqSet]#将一个频繁集中的每一个元素提取出来，单独作为一个列表，比如[0,1]会被分为[0]和[1]
            if (i > 1):#如果i>1表示一个频繁集中至少有3个元素，则后件可以取1个元素或2个元素，存在后件元素的合并
                rulesFromBackConseq(freqSet,H1,supportData,bigRuleList,minConf)#调用rulesFromConseq函数获取关联规则，那些规则会在函数中赋值到bigRuleList中
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList











#dataSet = loadDataSet()
# C1 = createC1(dataSet)
# D = map(set,dataSet)
# L1,supportData0 = scanD(D,C1,0.5)
# print L1
#L,support = apriori(dataSet)
#rule = generateRules(L,support)
#print rule
mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]#这样操作获取的是一行数据作为一个列表，每一个列表里面一个数据是一个元素
L,suppData = apriori(mushDataSet,minSupport=0.3)#生成一个列表，列表里的每一个元素也是一个列表，是一系列相同个数频繁集的集合
for item in L[1]:#遍历其中个数为2的频繁集
    if item & frozenset([2]):#将频繁集和2的元组求交集，若频繁集中存在2这个元素，则会输出
        print item