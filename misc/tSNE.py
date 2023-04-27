import os
import sys
import torch
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sns.set(style = "ticks", color_codes = True)

sys.path.append('../')

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


def color_lut():
    return {
        'F03': 'red',
        'F04': 'blue',
        'MC01': 'green',
        'MC04': 'yellow'
    }


def plot_t_SNE(x_vecs_path, perplexity=30):
    """ Function to perform t-SNE on x-vectors of a subject and visualize it on a lower dimensional space """

    file_names = sorted(os.listdir(x_vecs_path))
    xvecs = [torch.load(f"{x_vecs_path}/{file}")['xvector'] for file in file_names]
    
    # Performing t-SNE
    # tsne_feats = TSNE(2,perplexity=perplexity).fit_transform(np.array(xvecs))
    
    return np.array(xvecs), len(xvecs)


def main():
    total_xvecs, len_arr = [], []
    for sub in subs:
        sub_path = str(ROOTDIR + sub)
        xvecs, lens = plot_t_SNE(sub_path, perplexity = 30)
        total_xvecs.extend(xvecs)
        len_arr.append(lens)
    tsne_feats = TSNE(2,perplexity=30, verbose = 1).fit_transform(np.array(total_xvecs))
    for idx, sub in enumerate(subs):
        if idx == 0:
            plt.scatter(tsne_feats[:len_arr[0],0], tsne_feats[:len_arr[0],1], label = sub, c = color_lut()[sub], linewidths=0.1)
        elif idx == 1:
            plt.scatter(tsne_feats[len_arr[0]:len_arr[0]+len_arr[1],0], tsne_feats[len_arr[0]:len_arr[0]+len_arr[1],1], label = sub, c = color_lut()[sub], linewidths=0.1)
        elif idx == 2:
            plt.scatter(tsne_feats[len_arr[0]+len_arr[1]:len_arr[0]+len_arr[1]+len_arr[2],0], tsne_feats[len_arr[0]+len_arr[1]:len_arr[0]+len_arr[1]+len_arr[2],1], label = sub, c = color_lut()[sub], linewidths=0.1)
        else:
            plt.scatter(tsne_feats[len_arr[0]+len_arr[1]+len_arr[2]:len_arr[0]+len_arr[1]+len_arr[2]+len_arr[3],0], tsne_feats[len_arr[0]+len_arr[1]+len_arr[2]:len_arr[0]+len_arr[1]+len_arr[2]+len_arr[3],1], label = sub, c = color_lut()[sub], linewidths=0.1)
    # plt.scatter(tsne_feats[:,0], tsne_feats[:,1], label = sub, c = color_lut()[sub], linewidths=0.1, edgecolors='none')
    
    plt.legend(fontsize = 9, loc = 'best')
    plt.margins(0.05)
    plt.title('Visualizations of x-vector speaker embeddings using t-SNE for each speaker.', fontsize = 10, fontweight = 'bold')
    plt.savefig('tSNE_each_sub1.png')
    plt.show()

if __name__ == '__main__':
    ROOTDIR = './data/tmp/'
    subs = ['MC01', 'MC04', 'F03', 'F04']   
    main()
