import h5py
import numpy as np
from sklearn import svm

author = ['韦斯·安德森', '丹尼斯·维伦纽瓦', '马丁·斯科塞斯', '蒂姆·波顿', '王家卫', '爱丽丝·洛尔瓦彻']

f = h5py.File('film_data.mat', 'r')
tsdata = np.array(f.get('tsdata'))
scdata = np.array(f.get('scdata'))
lpdata = np.array(f.get('lpdata'))
sldata = np.array(f.get('sldata'))
cdata = np.array(f.get('cdata'))

X = np.transpose(np.concatenate((tsdata, scdata, lpdata, sldata, cdata), axis=0))
L = np.squeeze(np.array(f.get('L')))
print(X.shape, L.shape)


for k in range(1, 7):

    class_mask = L == k
    other_mask = L != k
    x_class = X[class_mask]
    x_other = X[other_mask]

    index = [x for x in range(x_class.shape[0])]
    np.random.shuffle(index)
    x_class = x_class[index]
    index = [x for x in range(x_other.shape[0])]
    np.random.shuffle(index)
    x_other = x_other[index]

    partinum = 6
    len_test = x_class.shape[0] // partinum
    recall_list_train = []
    precis_list_train = []
    recall_list_test = []
    precis_list_test = []

    for p in range(partinum):
        train_index_pre = np.arange(0, p * len_test)
        train_index_post = np.arange((p + 1) * len_test, x_class.shape[0])
        test_index = np.arange(p * len_test, (p + 1) * len_test)
        x_class_train = x_class[np.concatenate((train_index_pre, train_index_post), axis=0)]
        x_class_test = x_class[test_index]
        x_other_train = x_other[np.concatenate((train_index_pre, train_index_post), axis=0)]
        x_other_test = x_other[test_index]

        label_p = np.ones(x_class_train.shape[0])
        label_n = np.zeros(x_other_train.shape[0])
        if len(label_p) == 0:
            continue

        inputs = np.concatenate((x_class_train, x_other_train), axis=0)
        labels = np.concatenate((label_p, label_n), axis=0)

        svc = svm.SVC(C=1, gamma='scale', verbose=0, class_weight='balanced')
        svc.fit(inputs, labels)

        y_pred_p = svc.predict(x_class_train)
        y_pred_n = svc.predict(x_other_train)

        recall_train = np.mean(y_pred_p)
        presic_train = np.sum(y_pred_p) / (np.sum(y_pred_n) + np.sum(y_pred_p))
        recall_list_train.append(recall_train)
        if not np.isnan(presic_train):
            precis_list_train.append(presic_train)

        y_pred_p = svc.predict(x_class_test)
        y_pred_n = svc.predict(x_other_test)

        recall_test = np.mean(y_pred_p)
        presic_test = np.sum(y_pred_p) / (np.sum(y_pred_n) + np.sum(y_pred_p))
        recall_list_test.append(recall_test)
        if not np.isnan(presic_test):
            precis_list_test.append(presic_test)

    print('Author {:s}'.format(author[k - 1]))
    print('<TRAIN> precision:{:.2f} recall:{:.2f}'.format(np.mean(precis_list_train), np.mean(recall_list_train)))
    print('<TEST>  precision:{:.2f} recall:{:.2f}'.format(np.mean(precis_list_test), np.mean(recall_list_test)))
