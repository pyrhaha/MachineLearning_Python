#encoding:utf-8
#朴素贝叶斯的分类方法是通过根据贝叶斯条件概率，通过计算属于不同分类的概率，
# 如假设x1和y1分别表示两个特征，分类结果为c(i),则p( c(i) | x1,y1 )表示在当前特征取值下分类到不同类别的概率
# 而p( c(i) | x1,y1 ) = p( x1,y1 | c(i) ) * p( c(i) ) / p(x1,y1),
# p( x1,y1 | c(i) ) = p(x1|c(i)) * p(y1|c(i)), 通过计算训练集中分类为c(i)的数据集计算各个特征取值的概率
from numpy import *
import re
import feedparser
import operator

def loadDataSet():#创建实验样本
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1表示侮辱性言论，0表示正常言论
    return postingList,classVec

def createVocabList(dataSet):#该函数用于获取文档中出现的所有词,组成词汇表，不重复
    vocabSet = set([])#创建一个空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)#将各个数据集出现的词条求并集，从而得出在文档中出现过的所有词
    return  list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):#该函数用于形成文档向量，将出现的词置为1，两个参数为词汇表和某个文档
                                        #这个操作针对词集模型，即只对词在文档中出现与否作为特征，而不对词出现的数量进行统计
    returnVec = [0]*len(vocabList)#初始化各个词汇为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):#该函数用于形成文档向量，将出现的词置为1，两个参数为词汇表和某个文档
                                        #这个操作针对词袋模型,对词出现的数量进行统计
    returnVec = [0]*len(vocabList)#初始化各个词汇为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1#和词集模型唯一的区别在于
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec


def trainNB0(trainMatrix,trainCategory):#朴素贝叶斯分类器训练函数，输入参数为文档矩阵和由每篇文章类别标签所组成的向量
    numTrainDocs = len(trainMatrix)#记录文档的数量
    numWords = len(trainMatrix[0])#记录词汇表中词的数量
    pAbusive  = sum(trainCategory)/float(numTrainDocs)#sum的用处是将trainCategory中的所有数相加，因为若是侮辱性文档则值为1，
    # 所以sum的值为侮辱性文档的个数，除完结果为属于侮辱性文章的概率

    p0Num = ones(numWords)
    p1Num = ones(numWords)#词个数向量，初始化各词的个数为1,使用拉普拉斯平滑，为了防止由于该词出现的个数为0，从而使得用贝叶斯准则计算各概率时分母为零
    p0Denom = 2.0
    p1Denom = 2.0#表示分类为侮辱性文档中的词的总个数
    for i in range(numTrainDocs):#遍历每一个文档
        if trainCategory[i] == 1:#若该文档分类为侮辱性文档
            p1Num += trainMatrix[i]#因为出现的词该分量的值设置为1，所以进行向量的相加，出现的词的数量便增加1
            p1Denom += sum(trainMatrix[i])#将trainMatrix[i]的每一项相加即得到该文档词的总个数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom) #计算出的结果为在分类为侮辱性文档的所有文档中各词出现的概率
    p0Vect = log(p0Num/p0Denom) #对概率取对数是为了浮点数舍入导致的错误，由于
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):#贝叶斯分类函数（只针对二类分类问题），参数分别代表需要分类的词向量，p1Vec代表被分类为侮辱性文档的各词的概率对数向量
                                                 #p0Vec代表被分类为非侮辱性文档的各词的概率向量，pClass为分类为训练样本中侮辱性文档的概率
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)#vec2Classify与p1Vec两个向量相乘，由于vec2Classify中为1的分量与p1Vec相乘后表示概率，为0的分量则为0，最后求和，由于是对数，则表示各词分量的概率相乘后取对数
    p0 = sum(vec2Classify*p0Vec) + log(1 - pClass1)#由于是二类分类问题，所以1 - pClass1表示分类为非侮辱性文档的概率
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()#得到各文档的词列表和各文档的分类向量
    myVocabList = createVocabList(listOPosts)#得到模型的词列表，即各文档的词列表的并集
    trainMat = []
    for postinDoc in listOPosts:#遍历每篇文档的词列表
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))#将每篇文档中出现的词在对应的词向量中置为1
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classify as: ',classifyNB(thisDoc,p0V,p1V,pAb)



#垃圾邮件分类器
def textParse(bigString):#解析文档
    listOfTokens = re.split(r'\W*',bigString)#该正则表达式将非字母字符作为分隔符
    return [tok.lower() for tok in listOfTokens if len(tok)>2]#去掉其中一些URL分割后会有一些长度很小的词，将单词统一转为小写

def spamTest():
    docList = []#存储每个文档的词向量
    classList = []#存储标签的向量
    fullText = []#存储所有文档的词,用于高频词去除
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())#取出一个文档中的单词
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)#spam文件夹中存放的是侮辱性文章
        wordList = textParse(open('email/ham/%d.txt' % i).read())#取出一个文档中的单词
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)#ham文件夹中存放的是非侮辱性文章
    vocabList = createVocabList(docList)
    trainingSet = range(50)#共有50封邮件的数据集
    testSet = []
    for i in range(10):#随机选取10个数据集作为测试集,剩下的作为训练集
        randIndex = int(random.uniform(1,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:#创建训练集，以及对应的标签
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is:',float(errorCount)/len(testSet)


#RSS源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):#返回频率前30的单词
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortFreq = sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortFreq[:30]

def localWords(feed1,feed0):#对RSS源进行分类，并测试,这个函数和spamTest十分类似
    docList = []#存储每个文档的词向量
    classList = []#存储标签的向量
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])#将高频词从词汇表中删除
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(10):#随机选取10个数据集作为测试集,剩下的作为训练集
        randIndex = int(random.uniform(1,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:#创建训练集，以及对应的标签
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is:',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(ny,sf):#该函数是为了获得这篇文章最具代表性的词汇
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):#p0V和p1V长度相等
        if p0V[i] > -6.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key = lambda pair:pair[1],reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item
    sortedNY = sorted(topNY,key = lambda pair:pair[1],reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item



if __name__== '__main__':
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny,sf)