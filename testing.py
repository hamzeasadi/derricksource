import conf as cfg
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns






def plot_loss(path):
    with open(path, 'r') as f:
        filelines = f.readlines()
        print(filelines)



def cmtx(cm, classes, title,
                          normalize=False,
                          file='confusion_matrix',
                          cmap='gray_r',
                          linecolor='k'):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_title = 'Confusion matrix, with normalization'
    else:
        cm_title = title

    fmt = '.3f' if normalize else 'd'
    sns.heatmap(cm, fmt=fmt, annot=True, square=True,
                xticklabels=classes, yticklabels=classes,
                cmap=cmap, vmin=0, vmax=0,
                linewidths=0.5, linecolor=linecolor,
                cbar=False)
    sns.despine(left=False, right=False, top=False, bottom=False)

    plt.title(cm_title)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.savefig(f'{file}.png')
    # plt.show()


def main():
    x = np.random.randint(low=1, high=255, size=(4, 480, 800))
    # y = np.random.randint(low=1, high=255, size=(4, 480, 800))
    # fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[20, 6])
    # for j in range(4):
    #         axs[0, j].imshow(x[j])
    #         axs[0, j].axis('off')
    #         axs[1, j].imshow(y[j])
    #         axs[1, j].axis('off')

    # plt.subplots_adjust(wspace=0.1, hspace=0)
    # plt.show()
    plt.ion() 
    y = np.random.randint(low=0, high=6, size=(20, ))
    yhat = np.random.randint(low=0, high=6, size=(20, ))
    cnfmtx = confusion_matrix(y, yhat)
    # disp = ConfusionMatrixDisplay(cnfmtx)
    # plt.imshow(disp)
    # cb = plt.colorbar()
    # cb.remove()
    # plt.show()
    cls = [i for i in range(6)]
    cmtx(cm=cnfmtx, classes=cls, title='liebherr', file='liebherr')

    

if __name__ == '__main__':
    main()