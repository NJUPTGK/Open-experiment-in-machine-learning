import pandas as pd
import sklearn.tree as tree
from sklearn import preprocessing

dataset_path = 'D:\Python开放性实验\实验三\dataset for exp3.txt'
fr = open(dataset_path)
dataset_all_line = fr.readlines()

list_dataset = []  # 将数据集从txt专为list格式

lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']

lenses_target = []  # 数据集的结果列 list格式即可

pd_dataset = pd.DataFrame(columns=lenses_labels)  # pd_dataset存储dataframe格式的数据集

for line in dataset_all_line:  # 将dataset从txt转化为list
    first = line.split()[0:4]
    if len(line.split()) == 6:
        a = line.split()[4]
        b = line.split()[5]
        second = a + ' ' + b
    else:
        second = line.split()[4]
    first.append(second)
    lenses_target.append(second)
    list_dataset.append(first)

for i in range(len(list_dataset)):  # 将dataset从list转化为dataframe
    pd_dataset = pd_dataset.append(pd.DataFrame(
        {'age': list_dataset[i][0], 'prescript': list_dataset[i][1], 'astigmatic': list_dataset[i][2],
         'tearRate': list_dataset[i][3]}, index=[i]))

le = preprocessing.LabelEncoder()

for col in pd_dataset.columns:  # 将dataframe的每一列序列化 decisiontree只接受序列化的数据,具体百度
    pd_dataset[col] = le.fit_transform(pd_dataset[col])

# print(lenses_target)
# print(pd_dataset)
lenses_tree = tree.DecisionTreeClassifier(max_depth=5)  # 建立深度为5的tree模型
lenses_tree = lenses_tree.fit(pd_dataset, lenses_target)  # 训练

print(lenses_tree.predict([[1, 1, 0, 0],
                           [1, 1, 1, 1]]))  # 预测，可以改里面的值

"""
以下是txt数据集：
年龄  眼睛情况 是否散光 眼泪情况
young	myope	no	reduced	no lenses
young	myope	no	normal	soft
young	myope	yes	reduced	no lenses
young	myope	yes	normal	hard
young	hyper	no	reduced	no lenses
young	hyper	no	normal	soft
young	hyper	yes	reduced	no lenses
young	hyper	yes	normal	hard
pre	myope	no	reduced	no lenses
pre	myope	no	normal	soft
pre	myope	yes	reduced	no lenses
pre	myope	yes	normal	hard
pre	hyper	no	reduced	no lenses
pre	hyper	no	normal	soft
pre	hyper	yes	reduced	no lenses
pre	hyper	yes	normal	no lenses
presbyopic	myope	no	reduced	no lenses
presbyopic	myope	no	normal	no lenses
presbyopic	myope	yes	reduced	no lenses
presbyopic	myope	yes	normal	hard
presbyopic	hyper	no	reduced	no lenses
presbyopic	hyper	no	normal	soft
presbyopic	hyper	yes	reduced	no lenses
presbyopic	hyper	yes	normal	no lenses

"""









