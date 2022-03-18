import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = load_iris()
iris_X=iris.data
# print('特征变量的长度',len(iris_X))
iris_y = iris.target#目标值
# print('鸢尾花的目标值:',iris_y)
X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.2)
#利用train_test_split进行训练集和测试机进行分开，test_size占20%
from numpy import*
import operator
from decimal import Decimal

def myED(testdata,traindata):
    """ 计算欧式距离，要求测试样本和训练样本以array([ [],[],...[] ])的形式组织，
    每行表示一个样本，一列表示一个属性"""
    size_train=traindata.shape[0] # 训练样本量大小
    size_test=testdata.shape[0] # 测试样本大小
    XX=traindata**2
    sumXX=XX.sum(axis=1) # 行平方和
    YY=testdata**2
    sumYY=YY.sum(axis=1) # 行平方和
    Xpw2_plus_Ypw2=tile(mat(sumXX).T,[1,size_test])+\
    tile(mat(sumYY),[size_train,1])  #调用mat()函数可以将数组转换为矩阵
    EDsq=Xpw2_plus_Ypw2-2*(mat(traindata)*mat(testdata).T) # 欧式距离平方
    distances=array(EDsq)**0.5 #欧式距离
    print( "欧式距离" )
    return distances
print(myED(X_test,X_train))



from decimal import Decimal

def minkowski_distance(X, Y, p_value):
    def nth_root(value, n_root):
        root_value = 1 / float( n_root )
        return round( Decimal( value ) ** Decimal( root_value ), 3 )
    distance = []
    for i in range( 0, len( X ) ):
        for j in range( 0, len( Y ) ):
         distance.append( nth_root(pow(abs(Y[j][0]- X[i][0]),0.5),3)+  nth_root(pow(abs(Y[j][1]- X[i][1]),0.5),3)+ nth_root(pow(abs(Y[j][2]- X[i][2]),0.5),3)
                          +  nth_root(pow(abs(Y[j][3]- X[i][3]),0.5),3))
    distance = np.array( distance ).reshape( 30, 120 )
    print ("闵可夫斯基距离")
    return distance

print( minkowski_distance( X_train, X_test, 3 ) )

def manhadun_distance(x, y ):
    distance = []
    for i in range( 0, len( x ) ):
        for j in range( 0, len( y ) ):
            distance.append( abs( y[j][0] - x[i][0] ) + abs( y[j][1] - x[i][1] )
                     + abs( y[j][2] - x[i][2] ) + abs( y[j][3] - x[i][3] ) )
    print( "曼哈顿距离" )
    distance = np.array( distance ).reshape( 30, 120 )
    return distance
print( manhadun_distance( X_train, X_test ))
