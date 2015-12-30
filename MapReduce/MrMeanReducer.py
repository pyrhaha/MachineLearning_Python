#encoding:utf-8
#分布式计算均值和方差的MapReduce作业
#这是Reducer部分的代码
import sys
from numpy import *
def read_input(file):#将文件内容变成一个可迭代数据集
    for line in file:
        yield line.rstrip()#yield函数是一个生成器，原来若是使用列表来存储内存占用量太大，但是如果使用生成器生成的可迭代对象，
                           #在调用的时候每一次都会在yield处中断返回一个值，然后在下一次调用的时候继续执行，再在yield处中断
inputData = read_input(sys.stdin)#用于是使用hadoop流,可以用管道的方法将mapper的输出作为reducer的输入，所以输入的方式是标准流输入
mapperOut = [data.split('\t') for data in inputData]#这里使用for循环对inputData这个可迭代的对象进行迭代,得到的是mapper的输出
cumVal = 0.0#用于存储所有数据总和a+b
cumSumSq = 0.0#用于存储a^2+b^2
cumN = 0.0#用于存储数据的总个数
for instance in mapperOut:
    nj = float(instance[0])#instance的第0项内容为数据的个数
    cumN += nj
    cumVal += nj * float(instance[1])#由于第一项的内容为平均值，所以乘上数据的个数为数据总和
    cumSumSq += nj * float(instance[2])#第二项的内容是数据平方的均值，所以乘上nj得到的是a^2 + b^2
meanval = cumVal/cumN
varSum = (cumSumSq - 2 * meanval*cumVal + cumN * meanval*meanval)/cumN#计算方差，以两个数据来举例，方差为( (a - mean)^2 + (b - mean)^2 )/2将其拆开后就等于a^2+b^2 - 2*(a+b)*mean + n*mean^2
print '%d\t%f\t%f' % (cumN,meanval,varSum)
print >> sys.stderr , 'report: still alive'