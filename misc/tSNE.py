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

'''
def plot_t_SNE(x_vecs_hc, xvecs_p, tsne_dim=2, perplexity=100):
    # """ Function to perform t-SNE on x-vectors of a subject and visualize it on a lower dimensional space """
    # # creating an empty numpy array to append x-vecs for a subject
    # x_vec_subject=np.empty((1,x_vec_dim))
    # file_names = os.listdir(x_vecs_path)
    # # Going through x-vecs of a subject individually
    # for file in file_names:
    #     file_path = os.path.join(x_vecs_path, file)
    #     with open(file_path, 'r') as f:
    #         x_vec=np.loadtxt(f)
    #         x_vec_subject=np.append(x_vec_subject,[x_vec],axis=0)
    # x_vec_subject=x_vec_subject[1:,:]
    
    hc_length = x_vecs_hc.shape[0]
    p_length = xvecs_p.shape[0]

    total_xvecs = np.concatenate((x_vecs_hc, xvecs_p), axis = 0)

    # Performing t-SNE
    tsne=TSNE(tsne_dim,perplexity=perplexity, verbose=1)
    tsne_feats=tsne.fit_transform(total_xvecs)
    
    # Visualizing t-SNE transformed feats based on transformed feats' dim
    if tsne_dim==2:
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_feats[:hc_length,0], tsne_feats[:hc_length,1], label = 'Healthy Controls', c = 'green', linewidths=0.1)
        plt.scatter(tsne_feats[hc_length:,0], tsne_feats[hc_length:,1], label = 'Patients', c = 'red', linewidths=0.1)
        plt.title('Visualizations of x-vector speaker embeddings using t-SNE for healthy controls and patients.', fontsize = 12, fontweight = 'bold')
        plt.legend(fontsize = 10, loc = 'best')
        # reduce the margins
        plt.margins(0.05)
        plt.savefig('tSNE1.png')
        plt.show()
        # sns.scatterplot(x=tsne_feats[:,0],y=tsne_feats[:,1])
    elif tsne_dim==3:
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        ax.scatter3D(tsne_feats[:,0], tsne_feats[:,1], tsne_feats[:,2])
        plt.show()
    
    return


def main():
    xvec_hc, xvec_p = [], []
    for sub in subs:
        sub_path = str(ROOTDIR + sub)
        for file in sorted(os.listdir(sub_path)):
            xvec = torch.load(f"{sub_path}/{file}")['xvector']
            if 'C' in sub:
                xvec_hc.append(xvec)
            else:
                xvec_p.append(xvec)

    plot_t_SNE(np.array(xvec_hc), np.array(xvec_p), tsne_dim = 2, perplexity = 30)

'''


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