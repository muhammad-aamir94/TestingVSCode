import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_single_AUC(path_2_csv, pathology, algo_version):
    from sklearn import metrics
    data = pd.read_csv(path_2_csv, names=["Name", "prob", "label"])
    prob = data.prob.tolist()
    prob.pop(0)
    label = data.label.tolist()
    label.pop(0)
    arr_prob = np.array([float(i) for i in list(prob)])
    fpr, tpr, thresholds = metrics.roc_curve(np.array([int(i) for i in list(label)]),
                                            np.array([float(i) for i in list(prob)]))
    auc = metrics.roc_auc_score(np.array([int(i) for i in list(label)]),
                                            np.array([float(i) for i in list(prob)]))

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    dmin = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    idd = np.argmin(dmin)

    print('The Optimal threshold is: ', optimal_threshold)

    plt.plot(fpr, tpr, label="%s-Algo%s, auc=" % tuple([pathology, algo_version]) + '%.3f' % auc)
    plt.title('Opt Threshold: *: %.2f , O: %.2f'% tuple([optimal_threshold, thresholds[idd]]))
    plt.xlabel('1- Specificity ')
    plt.ylabel('Sensitivity ')
    plt.legend(loc=4)
    plt.scatter(fpr[np.int(optimal_idx)], tpr[np.int(optimal_idx)], c='r', marker='*')
    plt.scatter(fpr[idd], tpr[idd], c='b', marker='o')
    print(np.min(arr_prob))
    print(np.percentile(arr_prob, 99))
    plt.savefig(os.path.dirname(path_2_csv)+"/AUC.png")
    plt.show()

pathology = 'Atelectasis'
plot_single_AUC(r"Documents\AUC_Lesion.csv", 'Effusion', 'V8')
