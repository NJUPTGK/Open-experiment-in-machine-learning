from numpy import *
from os import listdir
import operator
# 输入数据为32*32的二进制图像矩阵
def img2vector(filename):
    returnVect = zeros((1, 1024))  # 返回一个1行1024列的矩阵，用0填充
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])  # 矩阵的转置，32*32转成1*1024
    return returnVect

def classifiy0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]     # 手写体样本集容量，inX就是1*1024的矩阵
    # (以下三行)距离计算,欧式距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # 沿Y轴复制dataSetSize倍，X轴复制一倍，减所有数据集的矩阵，就是欧式距离
    sqDiffMat = diffMat**2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 每一行求和，然后相加
    distances = sqDistances**0.5   # 欧氏距离开平方
    sortedDistIndicies = distances.argsort()  # x.argsort()是将X中的元素从小到大排序,提取其对应的index(索引)并输出
    classCount = {}
    # (以下两行)选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # 这里是选择距离最小的k个点， sortedDistIndicies已经排好序，只需迭代的取前k个样本点的labels(即标签)，
        # 并计算该标签出现的次数，这里还用到了dict.get(key, default=None)函数，key就是dict中的键voteIlabel，
        # classCount里如果不存在则返回一个0，如果存在则返回当前值并+1

    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse = True)
    #  operator.itemgetter(1)表示按照元组的第二个元素的次序对元组进行排序，倒序排列
    return sortedClassCount[0][0]
    #  返回识别出来的数字


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(path='trainingDigits')  # 获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))  # m行1024列的矩阵
    for i in range(m):
        # 一下三行，从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  # 所有的数据集都合在一个矩阵里
    testFileList = listdir(path='testDigits')

    errorCount = 0.0  # 错误个数计数器
    mTest = len(testFileList)

    # 从测试数据中提取数据
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]

        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classifiy0(vectorUnderTest, trainingMat, hwLabels, 3)  # hwLabels就是含有1-9的数字的列表，总数为m

        print("the classifier came back with:%d,the real answer is:%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0

    # 输出结果
    print("\nthe total number of errors is:%d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

handwritingClassTest()