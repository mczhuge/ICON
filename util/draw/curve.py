import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 'blue', 'green', 'purple', 'black', 'cyan', 'mistyrose', 'pink', 'gold', 'magenta', 'darkorange', 'dimgray', 'lime'


dataset    = ['ECSSD','PASCAL-S', 'DUTS', 'HKU-IS', 'DUT-OMRON']
algorithm  = ['RAS', 'PiCANet', 'AFNet', 'BASNet', 'CPD-R', 'F3Net', 'SCRN', 'EGNet-R', 'MINet-R', 'ITSD-R', 'GateNet-R', 'Ours-R']
colors     = ['blue', 'green', 'purple', 'gold', 'black', 'cyan', 'blue', 'green', 'purple', 'gold', 'black', 'red'] 
linestyles = ['-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '-']
linewidths = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.5]
axisPR     = [[0.0, 1.0, 0.7, 1.0], [0.0, 1.0, 0.65, 0.95], [0.0, 1.0, 0.675, 0.975], [0.0, 1.0, 0.7, 1.0], [0.0, 1.0, 0.6, 0.9]]
axisFT     = [[0.0, 255.0, 0.75, 0.96], [0.0, 255.0, 0.70, 0.89], [0.0, 255.0, 0.6,0.9], [0.0, 255.0, 0.7, 0.95], [0.0, 255.0, 0.60, 0.84]]

plt.subplots_adjust(top=0.95, left=0.04, bottom=0.06, right=0.995,  wspace =0.2, hspace =0.3)
flag = True
for i, data in enumerate(dataset):
    plt.subplot(2,5,1+i)
    for j, algo in enumerate(algorithm):
        recall    = sio.loadmat('./result/'+algo+'/'+data+'/rec.mat')['rec'][0]
        precision = sio.loadmat('./result/'+algo+'/'+data+'/prec.mat')['prec'][0]
        plt.plot(recall, precision, color=colors[j], linestyle=linestyles[j], linewidth=linewidths[j], label=algo)
        plt.axis(axisPR[i])
    #plt.title(data, fontsize=12)
    plt.xlabel('Recall', fontsize=14)
    if flag:
        flag = False
        plt.ylabel('Precision', fontsize=14)
    plt.grid(ls='--')
    plt.legend(loc="lower center", ncol=2, fontsize=9)


flag = True
for i, data in enumerate(dataset):
    plt.subplot(2,5,6+i)
    for j, algo in enumerate(algorithm):
        precision = sio.loadmat('./result/'+algo+'/'+data+'/prec.mat')['prec'][0]
        recall    = sio.loadmat('./result/'+algo+'/'+data+'/rec.mat')['rec'][0]
        F_measure = 1.3*precision*recall/(0.3*precision+recall)
        plt.plot(np.linspace(0, 256, 256), F_measure, color=colors[j], linestyle=linestyles[j], linewidth=linewidths[j], label=algo)
        plt.axis(axisFT[i])

    #plt.title(data, fontsize=12)
    plt.xlabel('Threshod', fontsize=14)
    if flag:
        flag = False
        plt.ylabel('F-measure', fontsize=14)
    plt.legend(loc="lower center", ncol=2, fontsize=9)
        
    plt.grid(ls='--')



plt.show()

