import numpy as np

def PSP_delta(feat, N):
    #https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat

Xind = np.arange(0,12,2)
Yind = np.arange(1,13,2)

def DeriveEMAfeats(EmaDataIp):
    #EMA: time X dim
    N = 4
    vel = PSP_delta(EmaDataIp,N)
    acc = PSP_delta(vel,N)
    Mvel = np.sqrt(np.square(vel[:,Xind]) + np.square(vel[:,Yind]))
    Macc = np.sqrt(np.square(acc[:,Xind]) + np.square(acc[:,Yind]))
    DerEMA = np.concatenate((EmaDataIp, Mvel, Macc),axis=-1) #static , velocity and acc combined for training. 
    return DerEMA

