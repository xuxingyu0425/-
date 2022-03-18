import numpy #曼哈顿距离
import math
import pdb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def KNN_arithm(iris_datasets, expected_data):  # 定义knn算法函数，函数参数为训练集和测试集
    iris_data, expected_data, iris_target, expected_target = train_test_split( iris_datasets['data'],
                                                                               iris_datasets['target'], random_state=0 )
    k = int( input( "请输入 k:" ) )  # 输入邻近点数
    test_target = []  # 用来存放测试结果
    for j in range( 0, len( expected_data ) ):
        distance = []  # 用来存放测试集数据与训练集数据的欧氏距离
        iris1 = []  # 用来存放测试集数据与第一类鸢尾花数据的曼哈顿距离
        iris2 = []  # 用来存放测试集数据与第二类鸢尾花数据的曼哈顿距离
        iris3 = []  # 用来存放测试集数据与第三类鸢尾花数据的曼哈顿距离
        index1 = 0  # 测试集数据与第一类鸢尾花邻近个数
        index2 = 0  # 测试集数据与第二类鸢尾花邻近个数
        index3 = 0  # 测试集数据与第三类鸢尾花邻近个数
        for i in range( 0, len( iris_data ) ):  # 计算曼哈顿距离
            distance.append( abs(expected_data[j][0] - iris_data[i][0])+ abs(expected_data[j][1] - iris_data[i][1])
                             +abs(expected_data[j][2] - iris_data[i][2])+ abs(expected_data[j][3] - iris_data[i][3]))
            if (iris_target[i] == 0):
                iris1.append( abs(expected_data[j][0] - iris_data[i][0])+ abs(expected_data[j][1] - iris_data[i][1])
                             +abs(expected_data[j][2] - iris_data[i][2])+ abs(expected_data[j][3] - iris_data[i][3]) )
            elif (iris_target[i] == 1):
                iris2.append( abs(expected_data[j][0] - iris_data[i][0])+ abs(expected_data[j][1] - iris_data[i][1])
                             +abs(expected_data[j][2] - iris_data[i][2])+ abs(expected_data[j][3] - iris_data[i][3]) )
            elif (iris_target[i] == 2):
                iris3.append( abs(expected_data[j][0] - iris_data[i][0])+ abs(expected_data[j][1] - iris_data[i][1])
                             +abs(expected_data[j][2] - iris_data[i][2])+ abs(expected_data[j][3] - iris_data[i][3]) )
            distance.sort( reverse=False )  # 将列表元素升序排列
            distance = distance[0:k - 1]  # 截取最短的k个距离
        for m in distance:  # 计算三个类别邻近数
            if m in iris1:
                index1 = index1 + 1
            elif m in iris2:
                index2 = index2 + 1
            else:
                index3 = index3 + 1
        final = [index1, index2, index3]
        final_index = final.index( max( final ) )
        if final_index == 0:
            test_target.append( 0 )
        elif final_index == 1:
            test_target.append( 1 )
        else:
            test_target.append( 2 )
    print( "预测分类：", test_target )
    correct = 0
    for i in range( 0, len( expected_target ) ):
        if (expected_target[i] == test_target[i]):
            correct = correct + 1
    print( "正确率为:", "%.2f%%" % (correct / len( expected_target ) * 100) )
    print("曼哈顿距离为：",distance)
if __name__ =="__main__":
    iris_datasets=load_iris()
    expected_data=[10,100,5,2]
KNN_arithm(iris_datasets, expected_data)