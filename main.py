#!/usr/bin/python3
import sys, getopt

from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from PIL import Image


def main(argv):
    featurefile = None
    personfile = None
    try:
        opts, args = getopt.getopt(argv, "hf:p:", ["ffile=", "pfile="])
    except getopt.GetoptError:
        print('python main.py -f <featurefile> -p <personfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python main.py -f <featurefile> -p <personfile>')
            sys.exit()
        elif opt in ("-f", "--ffile"):
            featurefile = arg
        elif opt in ("-p", "--pfile"):
            personfile = arg
    start = datetime.datetime.now()
    print('start')
    if not featurefile or not personfile:
        print('Не указаны файлы.')
        print('python main.py -f <featurefile> -p <personfile>')
        sys.exit(2)
    features = np.load(featurefile)
    person_id = np.load(personfile)
    print('Загрузка данных. Векторов: {0}, Персон: {1}. {2}'.format(len(features), len(person_id), timer(start)))

    y = []
    start_similarity = datetime.datetime.now()
    similarity = np.einsum('ij,kj->ik', features, features)
    print('Подсчет similarity готов. {0}'.format(timer(start_similarity)))

    start_meta = datetime.datetime.now()
    for index, sim in enumerate(similarity):
        prev = datetime.datetime.now()
        person = person_id[index]
        ii = np.where(person_id == person)[0]
        sub_sim = np.delete(sim, ii, axis=0)
        T = np.max(sub_sim)
        y.append(np.where(sim >= T, 1, 0))
        print('Вектор № {0}. Время: {1}'.format(index, datetime.datetime.now() - prev))
    print('Подсчет мета-данных готов. {0}'.format(timer(start_meta)))
    start_labels = datetime.datetime.now()
    true_labels = np.append([], y)
    print('Классы готовы. {0}'.format(timer(start_labels)))
    start_delete = datetime.datetime.now()
    del y[:]
    del y
    print('Удаляем мета. {0}'.format(timer(start_delete)))
    start_flatten = datetime.datetime.now()
    feature_similarity = similarity.flatten()
    print('Similarity готово. {0}'.format(timer(start_flatten)))

    start_axis = datetime.datetime.now()
    fpr, tpr, thresholds = roc_curve(true_labels, feature_similarity)
    print('frp tpr thresholds готово. {0}'.format(timer(start_axis)))
    start_roc_auc = datetime.datetime.now()
    roc_auc = auc(fpr, tpr)  # compute area under the curve
    print('Расчет площади графика готово. {0}'.format(timer(start_roc_auc)))
    start_plot= datetime.datetime.now()
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold', color='r')
    ax2.set_ylim([thresholds[-1], thresholds[0]])
    ax2.set_xlim([fpr[0], fpr[-1]])
    print('Генерация рафика готово. {0}'.format(timer(start_plot)))
    plt.savefig('roc_and_threshold_4.png')
    plt.close()
    print('Готово. Итого: {0}'.format(timer(start)))
    img = Image.open('roc_and_threshold_4.png')
    img.show()


def timer(start):
    return datetime.datetime.now() - start


if __name__ == "__main__":
   main(sys.argv[1:])