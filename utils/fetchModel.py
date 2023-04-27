import sys
sys.path.append('..')

from models.BLSTM import BLSTMNet

def get_model(cfg):
    if not cfg.use_feats:
        inputDim = cfg.nMFCC
    else:
        if cfg.feats in ['pase_plus']:
            inputDim = 256
        elif cfg.feats in ['vq_wav2vec', 'wav2vec', 'apc', 'npc']:
            inputDim = 512
        elif cfg.feats in ['audio_albert', 'tera', 'mockingjay', 'hubert']:
            inputDim = 768
        elif cfg.feats in ['decoar']:
            inputDim = 2048
        else:
            raise NotImplementedError(f"Feature {cfg.feats} not implemented")
        
    model = BLSTMNet(cfg, inputDim)
    model.init_model()
    return model, model.lstm_dim