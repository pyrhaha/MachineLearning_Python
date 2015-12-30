#encoding:utf-8
#该算法是SVD，将数据表示为A=U*S*(V.T),其中A为m*n的矩阵，U为m*k,S为k*k,V为k*n,这样k表示要将数据压缩成多少维度的，一般都是取最大的几个特征值
from numpy import *
from numpy import linalg as la

#加载数据
def loadData():
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


#使用欧氏距离计算相似度
def distanceSim(inA,inB):#计算公式为1/(1+ ( (a1-a2)^2 + (b1-b2)^2 + (c1-c2)^2 )开根号 )
    return 1.0/(1.0 + la.norm(inA-inB))#norm函数用于求范数，默认是2范数，就是将矩阵的每一个元素平方后相加，最后开根号

#使用皮尔逊相关系数来计算相似度
def pearsSim(inA,inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*corrcoef(inA,inB,rowvar=0)[0][1]

#使用余弦相似度算法
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5*(num/denom)

######################################################################

#标准的根据相似度来对评分进行预测，这是基于商品的推荐，即比如要预测某用户对商品A的评分，已知一些人对商品A和商品B的评分，通过计算这两个商品的相似度，就可以用该用户对B商品的评分来预测对A的评分
def standEst(dataMat,user,simMeas,item):#传入的参数有数据集，行数为用户的个数，列数为不同商品的个数，user是为需要预测的用户，simMeas为选用的计算相似度的方法，item为需要预测评分的商品
    n = shape(dataMat)[1]
    simTotal = 0.0
    rateSimTotal = 0.0
    for j in range(n):#遍历所有的商品
        userRate = dataMat[user,j]
        if userRate == 0:#即该用户没有对j商品评过分，则不能使用这个商品
            continue
        userChooseItemList = nonzero(logical_and(dataMat[:,item].A > 0,dataMat[:,j].A > 0))[0]#过滤出既对要预测的商品评分而且对j商品评分的那些用户
        if len(userChooseItemList) == 0:
            similarity = 0.0
        else:
            similarity = simMeas(dataMat[userChooseItemList,item],dataMat[userChooseItemList,j])#计算该商品与商品j之间的相似度
        simTotal += similarity#计算相似度的总和
        rateSimTotal += similarity*userRate#计算相似度和用户评分的总和
    if simTotal==0:
        return 0
    else:
        return rateSimTotal/simTotal#用评分总和和相似度总和将评分归一化到评分范围中

#推荐引擎，为用户推荐前N个评分预测最高的商品
def recommend(dataMat,user,N=3,simMeas = cosSim,estMethod = standEst):#传入的参数为数据集，用户的编号，N需要选择的前N个预测评分最高的商品，以及计算相似度的方法和计算预测评分的方法
    unratedItems = nonzero(dataMat[user,:] == 0)[1]#nonzero函数返回的是满足要求的项，是一个列表中有两个列表，第一个记录行数，第二个记录列数，这里需要的是商品的编号，所以是列数
    if len(unratedItems) == 0:
        return 'The user has rated everything'
    itemScores = []
    unratedItems = unratedItems.A[0]
    for item in unratedItems:
        estimateRate = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimateRate))
    return sorted(itemScores,key=lambda itemScores:itemScores[1],reverse=True)[:N]#将未评分的项目根据预测评分来排序，最后得到评分最大的前N个商品推荐给用户

#当数据集是一个稀疏矩阵时，先使用SVD进行分解，再用相似度算法完成推荐系统
def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    rateSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)#经过观察发现SVD后奇异值的前4个元素占总能量的90%，所以只取前4个
    Sig4 = mat(eye(4) * Sigma[:4])#eye函数生成一个4行4列的单位向量，然后与奇异值相乘，由于是numpy向量，所以相乘为元素相乘，即第一个矩阵的第一个元素和第二个矩阵的第二个元素相乘
    projItem = dataMat.T*U[:,:4]*Sig4.I#将原始数据的用户压缩到4列，一行表示一个商品，即所有的商品与这4列数据关系密切，此处求出的值其实就是V,证明过程在机器学习实战书P264
    for j in range(n):#与标准的推荐系统操作类似
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(projItem[item,:].T , projItem[j,:].T)
        print 'the similarity of %d and %d is: %f' % (item,j,similarity)
        simTotal += similarity
        rateSimTotal += similarity * userRating
    if similarity == 0:
        return 0
    else:
        return rateSimTotal/simTotal

######################################################
#基于SVD对图像进行压缩
#将图片打印出来
def printMat(inMat,thresh=0.8):#由于原数据是一个32*32的图片，且涉及灰度，是一个浮点型的数据，所以设置一个阈值
    for i in range(32):
        for j in range(32):
            if float(inMat[i,j]) > thresh:
                print 1,
            else:
                print 0,
        print ' '

#对图片进行压缩的主函数
def imgCompress(numSV = 3,thresh = 0.8):#传入的参数是压缩的SVD维度以及阈值thresh
    data = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        data.append(newRow)
    dataMat = mat(data)
    print '************original matrix***************'
    printMat(dataMat,thresh)
    U,Sigma,VT = la.svd(dataMat)
    Sig = mat(eye(numSV) * Sigma[:numSV])#将奇异值变成对角矩阵
    rebuildData = U[:,:numSV] * Sig * VT[:numSV,:]
    print '************rebuild matrix***************'
    printMat(rebuildData,thresh)



#data = mat(loadExData2())
#U,Sigma,VT = linalg.svd(data)
#print Sigma
#print U
#val,vec = linalg.eig(data*data.T)
#print vec
#Sig3 = mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])#因为奇异值中前三个数远大于后面的，所以取k为3来保存数据
#print U[:,:3]*Sig3*VT[:3,:]#重构原始矩阵的近似矩阵
#print distanceSim(data[:,0],data[:,4])
#print cosSim(data[:,0],data[:,4])

#print recommend(data,1,estMethod=svdEst)
imgCompress(2)