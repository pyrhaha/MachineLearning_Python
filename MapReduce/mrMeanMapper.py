#encoding:utf-8
#分布式计算均值和方差的MapReduce作业
#这是mapper部分的代码
import sys
from numpy import *
def read_input(file):#将文件内容变成一个可迭代数据集
    for line in file:
        yield line.rstrip()#yield函数是一个生成器，原来若是使用列表来存储内存占用量太大，但是如果使用生成器生成的可迭代对象，
                           #在调用的时候每一次都会在yield处中断返回一个值，然后在下一次调用的时候继续执行，再在yield处中断

inputData = read_input(sys.stdin)#用于是使用hadoop流，所以输入的方式是标准流输入
data = [float(data) for data in inputData]#这里使用for循环对inputData这个可迭代的对象进行迭代
numInputs = len(data)#获取需要求均值和方差的数据个数
data = mat(data)
sqData = power(data,2)#对矩阵中的每个元素求平方
print '%d\t%f\t%f' % (numInputs,mean(data),mean(sqData))#采用输出流的方式返回，返回的数据有(数据个数，数据的平均值，数据平方后的均值)
print >> sys.stderr, 'report: still alive'#这是向标准错误输出发送报告，一般默认的print其实是print >> sys.stdout

