import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append('../')

from src import *


PLOT = False

class Trainer():
    def __init__(
        self, args, model, loaders, device, fold
    ):
        self.args = args
        self.model = model
        self.loaders = loaders
        self.device = device
        self.fold = fold

        self.batchSize = self.args.batchSize
        self.epochs = self.args.noEpoch
        self.patience = self.args.patience
        self.lr = self.args.lr
        self.w_decay = self.args.weightDecay
        self.freq = self.args.freq
        self.verbose = self.args.verbose
        self.delta = self.args.delta
        self.minrun = self.args.minRun
        self.ft = self.args.fineTune
        self.units = self.args.noUnits

        self.counter = 0
        self.bestScore = None
        self.earlyStop = False
        self.valMinLoss = np.Inf
        self.best_cc_hc = [-1, 0]
        self.best_cc_p = [-1, 0]

        self.model.to(self.device)
        self.loss = AAILoss()

        self.use_feats = self.args.use_feats
        self.use_xvecs = self.args.use_xvec
        self.use_stats = self.args.use_stats

        self.cc_hc_total, self.cc_p_total = [], []
        self.rmse_hc_total, self.rmse_p_total = [], []

        self.emb = 'xvecs' if self.use_xvecs else 'stats'


    def train_loop(
        self, loader, mode = 'train'
    ):
      
        if mode == 'train':
            self.model.train()
        elif mode == 'val':
            self.model.eval()
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
        
        self.epoch_loss_dict = {}
        self.skipped = 0
        self.valLoss = []
        self.cc_hc, self.cc_p, self.rmse_hc, self.rmse_p = [], [], [], []
        self.std_hc, self.std_p = None, None

        with tqdm(loader, unit = 'batch', mininterval = 20) as tepoch:
            for batch_idx, data in enumerate(tepoch):
                if self.use_feats:
                    ema, feats, emb, ema_lens, labels, spks = data
                    ema, feats = ema.to(self.device), feats.to(self.device)
                else:
                    ema, mfcc, emb, ema_lens, labels, spks = data
                    ema, mfcc = ema.to(self.device), mfcc.to(self.device)
                # ema, mfcc, ema_lens, labels, spks = self.set_device(
                #     [ema, mfcc, ema_lens, labels, spks], ignoreList = []
                # )

                # ema, mfcc = ema.to(self.device), mfcc.to(self.device)

                # if np.argwhere(ema_lens == 0) is not None:
                #     self.skipped += 1
                #     idx_to_skip = np.argwhere(ema_len == 0)[0][0].item()
                #     ema = torch.cat((ema[:idx_to_skip], ema[idx_to_skip + 1:]), dim = 0)
                #     mfcc = torch.cat((mfcc[:idx_to_skip], mfcc[idx_to_skip + 1:]), dim = 0)
                #     ema_lens = torch.cat((ema_lens[:idx_to_skip], ema_lens[idx_to_skip + 1:]), dim = 0)
                #     labels = torch.cat((labels[:idx_to_skip], labels[idx_to_skip + 1:]), dim = 0)
                #     spks = torch.cat((spks[:idx_to_skip], spks[idx_to_skip + 1:]), dim = 0)

                # if len(ema_lens) != len((ema_lens != 0).nonzero()):
                #     self.skipped += 1
                #     idx_to_skip = (ema_lens != 0).nonzero()
                #     ema = ema[idx_to_skip]
                #     mfcc = mfcc[idx_to_skip]
                #     ema_lens = ema_lens[idx_to_skip]
                #     labels = labels[idx_to_skip]
                #     spks = spks[idx_to_skip]
                    
                if self.use_feats:
                    if self.use_xvecs or self.use_stats:
                        ema_pred = self.model(feats, emb.to(self.device))
                    else:
                        ema_pred = self.model(feats)
                else:
                    if self.use_xvecs or self.use_stats:
                        ema_pred = self.model(mfcc, emb.to(self.device))
                    else:
                        ema_pred = self.model(mfcc)
                    
                assert len(ema_pred) == len(ema), f"{len(ema_pred)} != {len(ema)}"

                loss_dict = self.loss.compute_loss(ema, ema_pred)
                
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss_dict['total'].backward()
                    self.optimizer.step()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                else:
                    healthy_control_ids = [idx for idx in range(len(spks)) if 'C' in spks[idx]]
                    patient_ids = [idx for idx in range(len(spks)) if 'C' not in spks[idx]]
                    # print(spks)
                    cc_hc, rmse_hc, cc_p, rmse_p = self.compute_cc(
                        ema.detach().cpu(), ema_pred.detach().cpu(), ema_lens.detach().cpu().numpy().astype(int).tolist(),
                        healthy_control_ids, patient_ids
                    )

                    if PLOT:
                        art_idx = int(self.articulatory_lut()[self.args.art_viz])
                        if cc_hc != [] and self.ft:
                            all_ccs = [cc_hc[idx][art_idx] for idx in range(len(cc_hc))]
                            idx_max = np.argmax(all_ccs)
                            gt_articulatory = ema[idx_max][:,art_idx].detach().cpu().numpy()
                            pred_articulatory = ema_pred[idx_max][:,art_idx].detach().cpu().numpy()
                            cc = cc_hc[idx_max][art_idx]
                            title = f'Healthy Controls - {self.args.art_viz} trajectory'
                            self.plot_trajectories(gt_articulatory, pred_articulatory, cc, title)
                        if cc_p != [] and self.ft:
                            all_ccs = [cc_p[idx][art_idx] for idx in range(len(cc_p))]
                            idx_max = np.argmax(all_ccs)
                            gt_articulatory = ema[idx_max][:,art_idx].detach().cpu().numpy()
                            pred_articulatory = ema_pred[idx_max][:,art_idx].detach().cpu().numpy()
                            cc = cc_p[idx_max][art_idx]
                            title = f'Patients - {self.args.art_viz} trajectory'
                            self.plot_trajectories(gt_articulatory, pred_articulatory, cc, title)

                    # self.cc.extend(cc)
                    # self.rmse.extend(rmse)

                    # self.cc_hc.append(cc_hc)
                    # self.cc_p.append(cc_p)
                    # self.rmse_hc.append(rmse_hc)
                    # self.rmse_p.append(rmse_p)

                    self.cc_hc.extend(cc_hc)
                    self.cc_p.extend(cc_p)
                    self.rmse_hc.extend(rmse_hc)
                    self.rmse_p.extend(rmse_p)

                    if self.args.infer:
                        self.cc_hc_total.extend(cc_hc)
                        self.cc_p_total.extend(cc_p)
                        self.rmse_hc_total.extend(rmse_hc)
                        self.rmse_p_total.extend(rmse_p)

                self.handle_metrics(loss_dict, mode = mode)
                tepoch.set_postfix(loss = loss_dict['total'].item())

        self.end_of_epoch(mode)
        
        # print(f"CC : {self.cc}, RMSE : {self.rmse}")
        if mode == 'val':
            print(f"CC (Healthy Controls) : {self.cc_hc}({(self.std_hc)}), RMSE (Healthy Controls): {self.rmse_hc}")
            print(f"CC (Patients) : {self.cc_p}({(self.std_p)}), RMSE (Patients) : {self.rmse_p}")
        return
        

    def end_of_epoch(self, mode):

        # path = 'tmp_files/file.txt'
        if mode == 'val':
            self.valLoss = sum(self.valLoss) / len(self.valLoss)

        for key in self.epoch_loss_dict:
            self.epoch_loss_dict[key] = sum(self.epoch_loss_dict[key]) / len(self.epoch_loss_dict[key])

        if mode == 'val':            
            if self.cc_hc != []:
                average_cc_hc = np.mean(self.cc_hc, axis = 0)
                print(f'Average CC (Healthy Controls) for articulators: {average_cc_hc}')

                self.std_hc = round(np.std(np.mean(self.cc_hc, axis = 0)), 3)
                self.cc_hc = round(np.mean(np.mean(self.cc_hc, axis = 0)), 4)
                self.rmse_hc = round(np.mean(np.mean(self.rmse_hc, axis = 0)), 4)

                if self.cc_hc > self.best_cc_hc[0]:
                    self.best_cc_hc = [self.cc_hc, self.curr_epoch]
                self.epoch_loss_dict['cc_hc'] = self.cc_hc

            if self.cc_p != []:
                average_cc_p = np.mean(self.cc_p, axis = 0)
                print(f'Average CC (Patients) for articulators: {average_cc_p}')

                self.std_p = round(np.std(np.mean(self.cc_p, axis = 0)), 3)
                self.cc_p = round(np.mean(np.mean(self.cc_p, axis = 0)), 4)
                self.rmse_p = round(np.mean(np.mean(self.rmse_p, axis = 0)), 4)

                if self.cc_p > self.best_cc_p[0]:
                    self.best_cc_p = [self.cc_p, self.curr_epoch]
                self.epoch_loss_dict['cc_p'] = self.cc_p


    def handle_metrics(
        self, loss_dict, mode = 'train'
    ):
        for key in loss_dict:
            if f"{key}_{mode}" not in self.epoch_loss_dict:
                self.epoch_loss_dict[f"{key}_{mode}"] = [loss_dict[key].item()]
            else:
                self.epoch_loss_dict[f"{key}_{mode}"].append(loss_dict[key].item())
        if mode == 'val':
            self.valLoss.append(loss_dict['aai'].item())


    def compute_cc(
        self, ema, ema_pred, ema_lengths, healthy_control_ids, patient_ids
    ):
        ema_ = ema.permute(0, 2, 1).numpy()
        ema_pred_ = ema_pred.permute(0, 2, 1).numpy()

        ema_hc = ema_[healthy_control_ids]
        ema_pred_hc = ema_pred_[healthy_control_ids]

        ema_p = ema_[patient_ids]
        ema_pred_p = ema_pred_[patient_ids]

        # cc_tmp, rmse_tmp = [], []
        cc_tmp_hc, rmse_tmp_hc = [], []
        cc_tmp_p, rmse_tmp_p = [], []

        ema_lengths_hc = [ema_lengths[idx] for idx in healthy_control_ids]
        ema_lengths_p = [ema_lengths[idx] for idx in patient_ids]

        for idx in range(len(ema_pred_hc)):
            c = []
            rmse = []
            for art in range(12):
                c.append(pearsonr(ema_hc[idx][art][:ema_lengths_hc[idx]], ema_pred_hc[idx][art][:ema_lengths_hc[idx]])[0])
                rmse.append(np.sqrt(np.mean(np.square(ema_hc[idx][art][:ema_lengths_hc[idx]] - ema_pred_hc[idx][art][:ema_lengths_hc[idx]]))))
            cc_tmp_hc.append(c)
            rmse_tmp_hc.append(rmse)


        for idx in range(len(ema_pred_p)):
            c = []
            rmse = []
            for art in range(12):
                c.append(pearsonr(ema_p[idx][art][:ema_lengths_p[idx]], ema_pred_p[idx][art][:ema_lengths_p[idx]])[0])
                rmse.append(np.sqrt(np.mean(np.square(ema_p[idx][art][:ema_lengths_p[idx]] - ema_pred_p[idx][art][:ema_lengths_p[idx]]))))
            cc_tmp_p.append(c)
            rmse_tmp_p.append(rmse)

        return cc_tmp_hc, rmse_tmp_hc, cc_tmp_p, rmse_tmp_p
    

    def check_early_stop(self):
        score = -self.valLoss
        if self.curr_epoch > self.minrun:
            if self.bestScore is None:
                self.bestScore = score
                self.saveckpt()
            elif score < self.bestScore + self.delta:
                self.counter += 1
                print(f"Early stopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    print("Early stopping")
                    self.earlyStop = True
            else:
                self.bestScore = score
                self.saveckpt()
                self.counter = 0


    def train(self):
        train, val, _ = self.loaders
        self.optimizer, self.scheduler = self.get_optimizer_scheduler()
        
        print('-' * 40)
        
        print('Training started')
        # if not self.args.infer:
        if self.ft:
            self.loadckpt(self.args.ft_ckpt)

        self.saveckpt(dry_run = True)

        for epoch in range(self.epochs + 1):
            print(f"Epoch {epoch} of {self.epochs}")

            self.curr_epoch = epoch
            self.train_loop(train, mode = 'train')
            self.train_loop(val, mode = 'val')

            self.scheduler.step(self.valLoss)

            self.check_early_stop()
            if self.earlyStop:
                print(f"Early stopping at epoch {epoch}")
                break
            
        print(f"Best cc for healthy controls: {self.best_cc_hc[0]} at epoch {self.best_cc_hc[1]}")
        print(f"Best cc for patients: {self.best_cc_p[0]} at epoch {self.best_cc_p[1]}")
        print('Training completed')
        # print('----------------------------------------')


    def infer(self):
        _, _, test = self.loaders

        print('----------------------------------------')
        print('Inference mode')
        self.curr_epoch = 1
        feat = self.args.feats if self.use_feats else 'mfcc'
        if self.args.no_subs == 'all':
            if self.args.pooled:
                if not self.use_xvecs and not self.use_stats:
                    ckpt = f"{self.args.ckpt}/pooled/LSTM_{self.units}_{feat}_{self.args.nEMA}_fold_{self.fold}.pth"
                else:
                    ckpt = f"{self.args.ckpt}/pooled/LSTM_{self.units}_{feat}_{self.emb}_{self.args.nEMA}_fold_{self.fold}.pth"
        elif self.args.no_subs == 1:
            folder = f"{self.args.ckpt}/{self.args.sub}"
            if self.ft:
                if not self.use_xvecs and not self.use_stats:
                    ckpt = f"{folder}/ft_<LSTM_{self.units}_{feat}_{self.args.nEMA}_fold_{self.fold}>_LSTM_{self.units}_{feat}_{self.args.nEMA}_fold_{self.fold}.pth"
                else:
                    ckpt = f"{folder}/ft_<LSTM_{self.units}_{feat}_{self.emb}_{self.args.nEMA}_fold_{self.fold}>_LSTM_{self.units}_{feat}_{self.emb}_{self.args.nEMA}_fold_{self.fold}.pth"
            else:
                if not self.use_xvecs and not self.use_stats:
                    ckpt = f"{folder}/LSTM_{self.units}_{feat}_{self.args.nEMA}_fold_{self.fold}.pth"
                else:
                    ckpt = f"{folder}/LSTM_{self.units}_{feat}_{self.emb}_{self.args.nEMA}_fold_{self.fold}.pth"
        else:
            raise NotImplementedError("No ckpt provided")

        print(f"Loading ckpt from {ckpt}")
        self.model.load_state_dict(
            torch.load(ckpt, map_location = 'cpu')
        )
        self.train_loop(test, mode = 'val')
        print(f"Test cc for healthy controls: {self.best_cc_hc[0]}")
        print(f"Test cc for patients: {self.best_cc_p[0]}")

        return self.cc_hc_total, self.cc_p_total, self.rmse_hc_total, self.rmse_p_total


    def get_optimizer_scheduler(self):
        model_params = [params for params in self.model.parameters() if params.requires_grad]

        if self.args.optimiser == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr = float(self.lr), weight_decay = float(self.w_decay))
        else:
            raise ValueError(f"Optimizer {self.args.optimiser} not supported")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.6, patience = 3, verbose = True)

        return optimizer, scheduler     
    

    def loadckpt(self, ckpt):
        if ckpt is None:
            if not self.ft:
                raise NotImplementedError("No finetune checkpoint provided")
            else:
                feat = self.args.feats if self.use_feats else 'mfcc'
                if not self.use_xvecs and not self.use_stats:
                    ckpt = f"{self.args.ckpt}/pooled/LSTM_{self.units}_{feat}_{self.args.nEMA}_fold_{self.fold}.pth"
                else:
                    ckpt = f"{self.args.ckpt}/pooled/LSTM_{self.units}_{feat}_{self.emb}_{self.args.nEMA}_fold_{self.fold}.pth"
                self.ft_ckpt = ckpt
        ckpt = torch.load(ckpt, map_location = self.device)
        self.model.load_state_dict(ckpt)
        # print(f"Loaded checkpoint from {ckpt}")


    def saveckpt(self, dry_run = False):
        if self.args.baseline:
            if self.use_feats:
                if not self.use_xvecs and not self.use_stats:
                    ckpt = f"LSTM_{self.units}_{self.args.feats}_{self.args.nEMA}_fold_{self.fold}"
                else:
                    ckpt = f"LSTM_{self.units}_{self.args.feats}_{self.emb}_{self.args.nEMA}_fold_{self.fold}"
            else:
                if not self.use_xvecs and not self.use_stats:
                    ckpt = f"LSTM_{self.units}_mfcc_{self.args.nEMA}_fold_{self.fold}"
                else:
                    ckpt = f"LSTM_{self.units}_mfcc_{self.emb}_{self.args.nEMA}_fold_{self.fold}"
        else:
            ckpt = '_'.join([
                f"loss_{self.args.loss}",
                f"batch_{self.batchSize}",
                f"{self.args.checkpoint_tag}"
            ])

        if self.args.no_subs == 1:
            if self.ft:
                ckpt = 'ft_<' + Path(self.ft_ckpt).stem +'>_' + ckpt
            folder = f"{self.args.ckpt}/{self.args.sub}"
            ckpt = f"{folder}/{ckpt}.pth"
            if not os.path.exists(folder):
                os.makedirs(folder)

        elif self.args.no_subs == 'all':
            if self.ft: raise NotImplementedError
            if not self.args.pooled: raise NotImplementedError
            folder = f"{self.args.ckpt}/pooled"
            ckpt = f"{folder}/{ckpt}.pth"
            if not os.path.exists(folder):
                os.makedirs(folder)

        else:
            raise NotImplementedError

        if not dry_run:
            torch.save(
                self.model.state_dict(), ckpt
            )
            self.valminloss = self.valLoss
        else:
            print(f"Checkpoint - {ckpt}")


    def set_device(self, data, ignoreList):
        if isinstance(data, list):
            return [
                data[idx].to(self.device).float() if idx not in ignoreList else data[idx] for idx in range(len(data))
            ]
        else:
            raise Exception('Set device for input not defined!')
        

    def plot_trajectories(self, gt, pred, cc = None, title = None):
        assert len(gt) == len(pred), "GT EMA and predicted EMA must be of the same length"
        plt.figure(figsize = (10, 10))
        plt.plot(gt, 'r', label = 'Ground Truth Articulatory Trajectory')
        plt.plot(pred, 'b', label = 'Predicted Articulatory Trajectory')
        plt.text(len(gt), gt[-1], f"CC: {cc:.3f}", fontsize = 12)
        plt.title(title)
        plt.legend()
        plt.show()


    def articulatory_lut(self):
        return {
            'ULx' : 0,
            'ULy' : 1,
            'LLx' : 2,
            'LLy' : 3,
            'Jawx' : 4,
            'Jawy' : 5,
            'TTx' : 6,
            'TTy' : 7,
            'TBx' : 8,
            'TBy' : 9,
            'TDx' : 10,
            'TDy' : 11
        }