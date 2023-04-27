import os
import sys
import time
import random
import librosa
import numpy as np
from tqdm import tqdm
import s3prl.hub as hub
from librosa.filters import mel 
from scipy import signal
from scipy.io import loadmat


import warnings
# warnings.filterwarnings("error")

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .processing import *

sys.path.append('../')
from utils.auxiliary import *
from utils.collate import *


class Loader(Dataset):
    def __init__(
        self, args, fold, mode = 'train'
    ):
        self.args = args
        self.mode = mode
        self.fold = fold
        self.datasetName = self.args.datasetName
        self.data_loc = f"{self.args.dataPath}/processed"
        self.tmp_dump = f"{self.args.dataPath}/tmp"
        self.splitRatio = self.args.trainSplit
        self.labels_list = self.get_all_labels()

        if self.mode == 'train':
            self.prepare_data()

        if self.args.no_subs == 'all': # pooled
            self.curr_subs = sorted(os.listdir(self.data_loc))
            if mode == 'train': print(f"Training scheme: Pooled | Subjects: {self.curr_subs}")
        elif isinstance(self.args.no_subs, int): # fine-tune
            if self.args.no_subs == 1:
                self.curr_subs = [self.args.sub]
                if self.args.fineTune:
                    if mode == 'train': print(f"Training scheme: Fine-tune | Subject: {self.curr_subs}")
                else:
                    if mode == 'train': print(f"Training scheme: Single subject | Subject: {self.curr_subs}")
        else:
            raise NotImplementedError

        self.split_based_on_mode()

        if self.mode == 'train' or self.mode == 'val':
            self.k_fold_split()

        # self.return_keys = {
        #     'label',
        #     'ema_norm',
        #     'emaLength',
        #     'mfcc_norm',
        #     'mfccLength',
        #     'log_mel',
        #     'melLength',
        #     'fs',
        #     'subject'
        # }

        self.return_keys = {
            'label',
            'ema_norm',
            'emaLength',
            'mfcc_norm',
            'mfccLength',
            'xvector',
            'BGEN',
            'subject'
        }

        if self.args.use_feats:
            self.get_ssl_paths()
        else:
            self.ssl_paths = None


    def get_ssl_paths(self):
        self.ssl_paths = {}
        ssl_dump = f"{self.args.dataPath}/ssl_feats/{self.args.feats}"
        for sub in self.curr_subs:
            for file in sorted(os.listdir(f"{ssl_dump}/{sub}")):
                if file.endswith('.pt'): self.ssl_paths[str(file)] = Path(file).stem


    def get_all_labels(self):
        all_labels = []
        for sub in sorted(os.listdir(self.data_loc)):
            for session in sorted(os.listdir(f"{self.data_loc}/{sub}")):
                sess_dir = f"{self.data_loc}/{sub}/{session}"
                for label in sorted(os.listdir(f"{sess_dir}/phn_headMic")):
                    all_labels.append(f"{sess_dir}/phn_headMic/{label}")
        return all_labels
    

    def split_based_on_mode(self):
        # final_files = []
        files = []
        for sub in self.curr_subs:
            # files = []
            data_loc = f"{self.tmp_dump}/{sub}"
            data_files = [f"{data_loc}/{file}" for file in sorted(os.listdir(data_loc))]

            # q = int(len(data_files) / self.args.folds)
            # total_n = int(q * self.args.folds)
            # data_x = data_files[:total_n]
            # splits = [
            #     [idx, idx + int(len(data_x) / self.args.folds)] for idx in range(0, len(data_x), int(len(data_x) / self.args.folds))
            # ]
            # if self.fold == 0:
            #     if mode == 'test':
            #         files.extend(data_x[splits[self.fold][0]:splits[self.fold][1]])
            #     else:
            #         files.extend(data_x[splits[self.fold][1]:])
            # elif self.fold in [1, 2, 3]:
            #     if mode == 'test':
            #         files.extend(data_x[splits[self.fold][0]:splits[self.fold][1]])
            #     else:
            #         files.extend(data_x[:splits[self.fold][0]])
            #         files.extend(data_x[splits[self.fold][1]:])
            # elif self.fold == 4:
            #     if mode == 'test':
            #         files.extend(data_x[splits[self.fold][0]:])
            #     else:
            #         files.extend(data_x[:splits[self.fold][0]])

            # if mode == 'train':
            #     final_files.extend(files[q:])
            # elif mode == 'val':
            #     final_files.extend(files[:q])
            # elif mode == 'test':
            #     final_files.extend(files)
            # else:
            #     raise NotImplementedError
            

            # random.shuffle(data_files)
            for idx, file in enumerate(data_files):
                if (idx + 10) % 10 == 0:
                    if self.mode == 'test':
                        files.append(file)
                # elif ((idx + 10 - 1) % 10) == 0:
                #     if mode == 'val':
                #         files.append(file)
                # else:
                #     if mode == 'train':
                #         files.append(file)
                else:
                    if self.mode == 'train' or self.mode == 'val':
                        files.append(file)

        # available_files = {Path(f).stem:f for f in get_files(self.tmp_dump, '.pt')}
        # self.files = [available_files[f] for f in files] 
        # print(f"Total of {len(self.files)} files for {mode}-ing")

        self.files = files
        if self.mode == 'test':
            print(f"Total of {len(self.files)} files for {self.mode}-ing")


    def k_fold_split(self):
        q = int(len(self.files) / self.args.folds)
        total_n = int(q * self.args.folds)
        data_x = self.files[:total_n]
        splits = [
            [idx, idx + int(len(data_x) / self.args.folds)] for idx in range(0, len(data_x), int(len(data_x) / self.args.folds))
        ]
        random.Random(4).shuffle(data_x)
        self.files = []
        if self.fold == 0:
            if self.mode == 'val':
                self.files.extend(data_x[splits[self.fold][0]:splits[self.fold][1]])
            else:
                self.files.extend(data_x[splits[self.fold][1]:])
        elif self.fold in [1, 2, 3]:
            if self.mode == 'val':
                self.files.extend(data_x[splits[self.fold][0]:splits[self.fold][1]])
            else:
                self.files.extend(data_x[:splits[self.fold][0]])
                self.files.extend(data_x[splits[self.fold][1]:])
        elif self.fold == 4:
            if self.mode == 'val':
                self.files.extend(data_x[splits[self.fold][0]:])
            else:
                self.files.extend(data_x[:splits[self.fold][0]])

        print(f"Total of {len(self.files)} files for {self.mode}-ing")


    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        data = [data[key] for key in data if key in self.return_keys]

        if self.ssl_paths is not None:
            feat = torch.load(str(self.files[idx]).replace('tmp', f'ssl_feats/{self.args.feats}'))
            data = data + [feat[-1]]

        return data
    
    
    def prepare_data(self):
        if not os.path.exists(self.tmp_dump):
            os.mkdir(self.tmp_dump)

        # if len(get_files(self.tmp_dump, '.pt')) == len(self.labels_list):
        if len(get_files(self.tmp_dump, '.pt')) != 0:
            print('All features have been generated and saved in the data/tmp folder.')
            return
        
        def fetch_mfcc(label):
            splits = label.split('/')
            mfcc_path = f"{self.args.dataPath}/processed/{splits[11]}/{splits[12]}/MFCC_Kaldi_wav_headMic/"
            filename = splits[-1].split('.')[0]
            return np.loadtxt(f"{mfcc_path}/{filename}.txt")
        
        def fetch_ema(label):
            splits = label.split('/')
            ema_path = f"{self.args.dataPath}/processed/{splits[11]}/{splits[12]}/EmaData/"
            
            filename = splits[-1].split('.')[0]

            #load ema(3d)
            ema = loadmat(f"{ema_path}/{filename}.mat")['ema']

            #filter and downsample
            b, a = signal.cheby2(10, 40, 40/(200/2))
            ema_200_filt = signal.filtfilt(b, a, ema, axis = 0)
            ema_100 = signal.decimate(ema_200_filt, 2, axis = 0) #time X 7 X 12

            ema_100_new = np.delete(ema_100, [1,3,4,5,6], 1) # time X 2 X 12
            ema_100_new = np.delete(ema_100_new, [3,4,8,9,10,11], 2) # time X 2 X 6(td,tb,tt,ul,ll,jaw)
        
            # tongue dorsum = tongue back
            # tongue middle = tongue body
            # lower incisor = jaw

            # changing order to {ul,ll,jaw,tt,tb,td} from {td,tb,tt,ul,ll,jaw}
            ul = np.concatenate((ema_100_new[:,0,3].reshape(-1,1), ema_100_new[:,1,3].reshape(-1,1)), axis = 1)
            ll = np.concatenate((ema_100_new[:,0,4].reshape(-1,1), ema_100_new[:,1,4].reshape(-1,1)), axis = 1)
            jaw = np.concatenate((ema_100_new[:,0,5].reshape(-1,1), ema_100_new[:,1,5].reshape(-1,1)), axis = 1)
            tt = np.concatenate((ema_100_new[:,0,2].reshape(-1,1), ema_100_new[:,1,2].reshape(-1,1)), axis = 1)
            tb = np.concatenate((ema_100_new[:,0,1].reshape(-1,1), ema_100_new[:,1,1].reshape(-1,1)), axis = 1)
            td = np.concatenate((ema_100_new[:,0,0].reshape(-1,1), ema_100_new[:,1,0].reshape(-1,1)), axis = 1)

            ema = np.concatenate((ul, ll, jaw, tt, tb, td), axis = 1)

            return ema
        

        def norm_mfcc_ema(
                m_t, ema, label
        ):
            try:
                MeanOfData = np.mean(ema, axis = 0) 
                Ema_temp2 = ema - MeanOfData
                C = 0.5 * np.sqrt(np.mean(np.square(Ema_temp2), axis = 0))  
                Ema = np.divide(Ema_temp2, C) # Mean remov & var normalised
                Ema = DeriveEMAfeats(Ema)
            except RuntimeError:
                print(f"Error in EMA for {label}")
                raise Exception

            MeanOfData=np.mean(m_t, axis = 0) 
            m_t -= MeanOfData
            C = 0.5 * np.sqrt(np.mean(np.square(m_t), axis = 0))
            mfcc = np.divide(m_t,C)

            ema_final, mfcc_final = [], []

            ema_temp = np.array([], dtype = np.float32).reshape(0, self.args.nEMA)
            mfcc_temp = np.array([], dtype = np.float32).reshape(0, self.args.nMFCC)

            BGEN = []

            with open(label) as f:
                labels = f.readlines()
                for d in labels:
                    start_index = int(int(d.split()[0]) / 160)
                    end_index = int(int(d.split()[1]) / 160)
                    if d.split(' ')[2] != 'noi\n' and d.split(' ')[2] != 'sil\n':
                        ema_temp = np.vstack((ema_temp, Ema[start_index:end_index,:]))
                        mfcc_temp = np.vstack((mfcc_temp, mfcc[start_index:end_index,:]))
                        BGEN.append((d.split()[0], d.split()[1]))

            # # Make chunks after taking in all phonemes.
            # if ema_temp.shape[0] > self.args.padMax:
            #     s, end = 0, self.args.padMax
            #     while end < ema_temp.shape[0]:
            #         ema_final.append(ema_temp[s:end, :])
            #         mfcc_final.append(mfcc_temp[s:end, :])
            #         if ema_temp.shape[0] - end < self.args.padMax:
            #             pad_zeroes_length = self.args.padMax - (ema_temp.shape[0] - end)
            #             e = np.vstack(
            #                 (ema_temp[end::], np.zeros((pad_zeroes_length, self.args.nEMA), dtype = float))
            #             )
            #             m = np.vstack(
            #                 (mfcc_temp[end::], np.zeros((pad_zeroes_length, self.args.nMFCC), dtype = float))
            #             )
            #             ema_final.append(e)
            #             mfcc_final.append(m)                 
            #         s, end = end, end + self.args.padMax
            # else:
            #     ema_final.append(np.vstack(
            #         (ema_temp, np.zeros((self.args.padMax - ema_temp.shape[0], self.args.nEMA), dtype = float))
            #     ))
            #     mfcc_final.append(np.vstack(
            #         (mfcc_temp, np.zeros((self.args.padMax - mfcc_temp.shape[0], self.args.nMFCC), dtype = float))
            #     ))

            ema_final, mfcc_final = ema_temp, mfcc_temp 
            return np.array(ema_final), np.array(mfcc_final), BGEN

        def fetch_speech_features(label):
            splits = label.split('/')
            wav_path = f"{self.args.dataPath}/processed/{splits[11]}/{splits[12]}/wav_headMic/"
            filename = splits[-1].split('.')[0]
            wav, fs = librosa.load(wav_path + filename + '.wav', sr = self.args.wavsampleRate)
            # fs, wav = wavfile.read(wav_path + filename + '.wav')
            wav_temp = np.array([], dtype = np.float32).reshape(0, 1)

            wav_final = []

            with open(label) as f:
                labels = f.readlines()
                for d in labels:
                    start_index = int(d.split()[0])
                    end_index = int(d.split()[1])
                    if d.split(' ')[2] != 'noi\n' and d.split(' ')[2] != 'sil\n':
                        wav_temp = np.vstack((wav_temp, wav[start_index:end_index].reshape(-1,1)))

            # # Make chunks after taking in all phonemes.

            # if wav_temp.shape[0] > self.args.padMax:
            #     s, end = 0, self.args.padMax * fs
            #     while end < wav_temp.shape[0]:
            #         wav_final.append(wav_temp[s:end])
            #         if wav_temp.shape[0] - end < self.args.padMax:
            #             pad_zeroes_length = self.args.padMax - (wav_temp.shape[0] - end)
            #             w = np.vstack(
            #                 (wav_temp[end::], np.zeros((pad_zeroes_length, 1), dtype = float))
            #             )
            #             wav_final.append(w)                   
            #         s, end = end, end + (self.args.padMax) * fs
            # else:
            #     wav_final.append(np.vstack(
            #         (wav_temp, np.zeros((self.args.padMax - wav_temp.shape[0], 1), dtype = float))
            #     ))

            # wav = torch.from_numpy(np.array(wav_final))
            y = torch.from_numpy(np.array(wav_temp))
            y = torch.clamp(y, min = -1, max = 1).numpy()

            spec = np.abs(
                librosa.stft(
                    y, n_fft = self.args.nfft, hop_length = self.args.hopLength, win_length = self.args.winLength
                )
            )

            mel = librosa.feature.melspectrogram(
                S = spec, sr = self.args.wavsampleRate, n_fft = self.args.nfft, n_mels = self.args.nMels, fmin = self.args.fMin, fmax = self.args.fMax
            ).T

            return mel, np.log(np.clip(mel, a_min = 1e-5, a_max = None)), fs
        
        def get_xvecs(label):
            splits = label.split('/')
            xvec_path = f"{self.args.dataPath}/processed/{splits[11]}/{splits[12]}/Xvector_Kaldi_headMic/"
            filename = splits[-1].split('.')[0]
            xvec = np.loadtxt(xvec_path + filename + '.txt')
            return xvec

        def extract_features(label):
            ema = fetch_ema(label)
            mfcc = fetch_mfcc(label)
            ema_norm, mfcc_norm, BGEN = norm_mfcc_ema(mfcc, ema, label)
            # mel, log_mel, fs = fetch_speech_features(label)

            if ema_norm.size != 0 and mfcc_norm.size != 0:
                subject = label.split('/')[11]
                save_name = f"{label.split('/')[12]}_{label.split('/')[-1].split('.')[0]}"

                # data = {
                #     'label': label,
                #     'ema_norm': ema_norm,
                #     'emaLength': ema_norm.shape[0],
                #     'mfcc_norm': mfcc_norm,
                #     'mfccLength': mfcc_norm.shape[0],
                #     'mel': mel,
                #     'log_mel': log_mel,
                #     'melLength': log_mel.shape[0],
                #     'fs': fs,
                #     'subject': subject
                # }

                data = {
                    'label': label,
                    'ema_norm': ema_norm,
                    'emaLength': ema_norm.shape[0],
                    'mfcc_norm': mfcc_norm,
                    'mfccLength': mfcc_norm.shape[0],
                    'xvector': get_xvecs(label),
                    'BGEN': BGEN,
                    'subject': subject
                }

                if not os.path.exists(f"{self.tmp_dump}/{subject}"):
                    os.makedirs(f"{self.tmp_dump}/{subject}")

                torch.save(
                    data,
                    f"{self.tmp_dump}/{subject}/{save_name}.pt"
                )
                return
            else:
                print(f"Error in {label}. Empty EMA or MFCC. Skipping...")
                self.labels_list.remove(label)
                return
        

        for label_path in tqdm(self.labels_list):
            extract_features(label_path)



def load_dataloaders(args, device, fold):
    loaders = []

    collate_fn = CollateClass(args)

    if args.use_feats and args.dump_feats:
        ssl_dump_location = f"{args.dataPath}/ssl_feats/{args.feats}"
        if os.path.isdir(ssl_dump_location):
            print(f"SSL features have already been dumped in data/ssl_feats/{args.feats}.")
        else:
            dump_ssl_feats(args, device)

    for mode in ['train', 'val', 'test']:
        dataset = Loader(args, fold, mode)
        dataloader = DataLoader(dataset,
                                batch_size = int(args.batchSize) if mode!='test' else 1,
                                shuffle = True if mode == 'train' else False,
                                num_workers = args.workers,
                                pin_memory = True,
                                collate_fn = collate_fn
                                )
        loaders.append(dataloader)
    return loaders


def dump_ssl_feats(args, device):
    print("Dumping SSL features...")
    files = get_files(f"{args.dataPath}/tmp", '.pt')

    def get_wavfiles(files):
        wav_files = []
        for file in sorted(files):
            path = str(file)
            splits = path.split('/')
            spk = splits[-2]
            session = splits[-1].split('_')[0]
            cnt = splits[-1].split('_')[1].split('.')[0]
            wav_files.append(
                f"{args.dataPath}/processed/{spk}/{session}/wav_headMic/{cnt}.wav"
            )
        assert len(wav_files) == len(files), "Number of wav files and labels don't match!"
        return wav_files

    wav_files = get_wavfiles(files)

    try:
        model = getattr(hub, str(args.feats))()
    except:
        raise Exception(f"Verify {args.feats} for pretrained s3prl")
    
    model = model.to(device)

    ssl_folder = f"{args.dataPath}/ssl_feats/{args.feats}"
    if not os.path.exists(ssl_folder):
        os.makedirs(ssl_folder)

    for wav in tqdm(wav_files):
        y, sr = librosa.load(wav, sr = args.wavsampleRate)
        y = torch.from_numpy(y).to(device)
        with torch.no_grad():
            reps = model([y])['hidden_states']
        reps = torch.cat(reps, 0).cpu()
        spk = wav.split('/')[-4]
        session = wav.split('/')[-3]
        save_path = f"{ssl_folder}/{spk}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cnt = wav.split('/')[-1].split('.')[0]
        torch.save(reps, f"{save_path}/{session}_{cnt}.pt")
    
