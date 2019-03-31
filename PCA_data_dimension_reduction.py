# PCA
import warnings
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
# 忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")

lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)  # resize用于调整每张脸部图片的大小,默认为0.5
# 提取的数据集将仅保留至少具有min_faces_per_person个不同图片的人的图片
n_samples, h, w = lfw_people.images.shape
x = lfw_people.data
n_features = x.shape[1]  # 返回有多少列的特征
y = lfw_people.target
target_names = lfw_people.target_names  # 目标的名字，返回一个包含人名的列表

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)  # 测试集比例是40%
# 先训练PCA模型
# n_components:意义：PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
# 类型：int 或者 string，缺省时默认为None，所有成分被保留。
# #赋值为int，比如n_components=1，将把原始数据降到一个维度。
#  赋值为string，比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。
PCA = PCA(n_components=100).fit(X_train)
# 返回测试集和训练集降维后的数据集
# fit()可以说是scikit-learn中通用的方法，
# 每个需要训练的算法都会有fit()方法，它其实就是算法中的“训练”这一步骤。因为PCA是无监督学习算法，此处y自然等于None。
# fit(X)，表示用数据X来训练PCA模型。
# 函数返回值：调用fit方法的对象本身。比如pca.fit(X)，表示用X对pca这个对象进行训练
# 识别测试集中的人脸
eigenfaces = PCA.components_.reshape((100, h, w))
x_train_pca = PCA.transform(X_train)
x_test_pca = PCA.transform(X_test)
# transform(X)
# 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维
# KNN核心代码
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train_pca, y_train)  # 用训练集进行训练模型

y_test_predict = knn.predict(x_test_pca)

'''
#输出
for i in range(len(y_test_predict)):
    print(target_names[y_test_predict[i]])

'''

print(knn.score(x_test_pca, y_test))  # 返回给定测试数据和标签的平均准确度
# print(metrics.classification_report(y_test, y_test_predict))  # 包含准确率，召回率等信息表
# print(metrics.confusion_matrix(y_test, y_test_predict))  # 混淆矩阵
def plot_gallery(images, titles, h, w, n_row=4, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_test_predict, y_test, target_names, i)
                     for i in range(y_test_predict.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()



