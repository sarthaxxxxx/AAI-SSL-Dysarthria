import numpy as np


import torch



class CollateClass():
    def __init__(
        self, args
    ):
        self.args = args
        self.ssl_feats = self.args.use_feats


    def __call__(self, batch):
        ema_lengths, idx_sorted_decreasing = torch.sort(
            torch.tensor([len(x[1]) for x in batch]),
            dim = 0,
            descending = True
        )

        max_ema_length = ema_lengths[0]
        ema_padded = torch.FloatTensor(len(batch), max_ema_length, self.args.nEMA)
        ema_padded.zero_()
        mfcc_padded = torch.FloatTensor(len(batch), max_ema_length, self.args.nMFCC)
        mfcc_padded.zero_()
        xvec_padded = torch.FloatTensor(len(batch), max_ema_length, self.args.nxvec)
        # xvec_padded.zero_()
        
        if self.ssl_feats:
            dim = batch[idx_sorted_decreasing[0]][-1].shape[-1]
            feat_padded = torch.FloatTensor(len(batch), max_ema_length, dim)
            feat_padded.zero_()

        stats_dim = 2 * self.args.nMFCC if not self.ssl_feats else 2 * dim
        stats_padded = torch.FloatTensor(len(batch), max_ema_length, stats_dim)
        # max_mel_len = max([x[5].shape[0] for x in batch])
        # mel_padded = torch.FloatTensor(len(batch), max_mel_len, self.args.nMels)
        # mel_padded.zero_()

        # ema_lens_, mel_lens_, spks, labels = [], [] ,[], [], [], []
        ema_lens_, spks, labels = [], [] ,[]

        for idx in range(len(idx_sorted_decreasing)):
            label = batch[idx_sorted_decreasing[idx]][0]
            labels.append(label)

            ema = batch[idx_sorted_decreasing[idx]][1]
            ema_padded[idx, :ema.shape[0], :] = torch.from_numpy(ema)
            ema_lens = batch[idx_sorted_decreasing[idx]][2]
            ema_lens_.append(ema_lens)

            mfcc = batch[idx_sorted_decreasing[idx]][3]
            mfcc_padded[idx, :mfcc.shape[0], :] = torch.from_numpy(mfcc)
            
            # mel = batch(idx_sorted_decreasing[idx])[5]
            # mel_padded[idx, :mel.shape[0], :] = torch.from_numpy(mel)
            # mel_lens = batch(idx_sorted_decreasing[idx])[6]
            # mel_lens_.append(mel_lens)
            xvec = batch[idx_sorted_decreasing[idx]][5]
            xvec_padded[idx, :, :] = torch.from_numpy(np.repeat(np.array(xvec)[np.newaxis, np.newaxis, :], max_ema_length, axis = 1))

            spk = batch[idx_sorted_decreasing[idx]][7]
            spks.append(spk)

            if self.ssl_feats:
                feat = batch[idx_sorted_decreasing[idx]][-1]
                BGEN = batch[idx_sorted_decreasing[idx]][6]
                feat_tmp = np.array([], dtype = np.float32).reshape(0, dim)
                for (start, stop) in BGEN:
                    start = int(int(start) / 160)
                    stop = int(int(stop) / 160)
                    feat_tmp = np.vstack((feat_tmp, feat.numpy()[start:stop, :]))
                # if feat_tmp.shape[0] != ema.shape[0]:
                #     feat_tmp = feat_tmp[:ema.shape[0], :]
                if feat_tmp.shape[0] > ema.shape[0]:
                    feat_tmp = feat_tmp[:ema.shape[0], :]
                elif feat_tmp.shape[0] < ema.shape[0]:
                    feat_tmp = np.vstack((feat_tmp, np.zeros((ema.shape[0] - feat_tmp.shape[0], dim))))
                else:
                    pass
                feat_padded[idx, :ema.shape[0], :] = torch.from_numpy(feat_tmp)

                stats = np.concatenate([np.mean(feat_tmp, axis = 0), np.std(feat_tmp, axis = 0)], axis = 0).transpose()
                stats_padded[idx, :, :] = torch.from_numpy(np.repeat(np.array(stats)[np.newaxis, np.newaxis, :], max_ema_length, axis = 1))
            else:
                stats = np.concatenate([np.mean(mfcc, axis = 0), np.std(mfcc, axis = 0)], axis = 0).transpose()
                stats_padded[idx, :, :] = torch.from_numpy(np.repeat(np.array(stats)[np.newaxis, np.newaxis, :], max_ema_length, axis = 1))

        if self.ssl_feats:   
            if self.args.use_stats:
                return ema_padded, feat_padded, stats_padded, t(ema_lens_), labels, spks
            else:
                return ema_padded, feat_padded, xvec_padded, t(ema_lens_), labels, spks
        else:
            if self.args.use_stats:
                return ema_padded, mfcc_padded, stats_padded, t(ema_lens_), labels, spks
            else:
                return ema_padded, mfcc_padded, xvec_padded, t(ema_lens_), labels, spks


            # return ema_padded, mfcc_padded, mel_padded, t(ema_lens_), t(mel_lens_), t(labels), t(spks)


def t(arr):
    if isinstance(arr, list):
        return torch.from_numpy(np.array(arr))
    elif isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)
    else:
        raise NotImplementedError