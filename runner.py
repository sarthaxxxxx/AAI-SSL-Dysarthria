import os
import sys
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from src import *
from utils import *
from trainer import *


def main(args, gpu):
    assert isinstance(args, object)
    args = Configuration(args)

    if gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        device = f"{gpu}" if torch.cuda.is_available() else 'cpu'

    if device is not None:
        torch.cuda.set_device(device)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    print(
        f"Device {device} is being used. | "
        f"Number of GPUs available: {torch.cuda.device_count()}."
    )

    mean_cc_hc, mean_cc_p, mean_rmse_hc, mean_rmse_p = [], [], [], []

    for fold in range(args.folds):
        print(f"Fold: {fold + 1}/{args.folds}")

        loaders = load_dataloaders(args, device, fold)
        print(
            f"Loaded the train, validation and test dataloaders. | "
            f"Pre-trained SSL model used: {args.feats}" if args.use_feats else f"Training using MFCCs."
        )

        try:
            network, inputDim = get_model(args)
            if not args.use_xvec and not args.use_stats:
                emb_msg = "No embeddings conditioned."
            else:
                if args.use_xvec:
                    emb_msg = "X-vectors conditioned to acoustic feats."
                elif args.use_stats:
                    emb_msg = "Mean and std conditioned to acoustic feats."
                else:
                    raise ValueError("Invalid embedding type.")

            print(
                f"Loaded the model. | "
                f"Model architecture: {args.model} | "
                f"Number of parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad)} | "
                f"Input dimension: {inputDim} | "
                f"{emb_msg} | "
            )

        except:
            print("Error loading the model. Exiting...")
            sys.exit(0)

        try:
            trainer = Trainer(
                args, network, loaders, device, fold
            )
            print(
                f"Initialized the trainer. | "
                f"Number of epochs: {args.noEpoch} | "
                f"Batch size: {args.batchSize} | "
                f"Learning rate: {args.lr} | "
                f"Optimizer: {args.optimiser} | "
                f"Loss function: {args.loss}"
            )
            
            if not args.infer:
                trainer.train()
            else:
                cc_hc, cc_p, rmse_hc, rmse_p = trainer.infer()
                mean_cc_hc.extend(cc_hc)
                mean_cc_p.extend(cc_p)
                mean_rmse_hc.extend(rmse_hc)
                mean_rmse_p.extend(rmse_p)

        except:
            print("Error training/infering. Exiting...")
            sys.exit(0)

        print(f"Finished fold {fold + 1}/{args.folds}.")

    print('-' * 40)

    if mean_cc_hc != [] and mean_rmse_hc != []:
        mean_hc_arti = np.mean(mean_cc_hc, axis = 0)
        mean_hc = np.mean(mean_hc_arti, axis = 0)
        print(f"Healthy Control : average CC for each articulator : {np.round(mean_hc_arti, 4)}")
        print(
            f"Healthy Controls: Mean CC: {round(mean_hc, 4)} | STD: {round(np.std(mean_hc_arti), 4)} | "
            f"RMSE: {round(np.mean(np.mean(mean_rmse_hc, axis = 0)), 4)}" 
        )
    if mean_cc_p != [] and mean_rmse_p != []:
        mean_p_arti = np.mean(mean_cc_p, axis = 0)
        mean_p = np.mean(mean_p_arti, axis = 0)
        print(f"Patient : average CC for each articulator : {np.round(mean_p_arti, 4)}")
        print(
            f"Patients: Mean CC: {round(mean_p, 4)} | STD: {round(np.std(mean_p_arti), 4)} | "
            f"RMSE: {round(np.mean(np.mean(mean_rmse_p, axis = 0)), 4)}" 
        )

    print('-' * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'CLI args for training and inference')
    parser.add_argument('--config', type = str,
                        default = 'configs/params.yaml')
    parser.add_argument('--gpu', type = str, default = 'cuda:0',
                        help = 'GPU to use')
    args = parser.parse_args()
    ROOTDIR = os.path.dirname(os.path.abspath(__file__))
    main(f"{ROOTDIR}/{args.config}", args.gpu)
