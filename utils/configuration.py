import yaml

class Configuration(object):
    def __init__(self, cfg, mode = 'params'):
        with open(cfg, 'r') as f:
            docs = yaml.load_all(f, Loader = yaml.FullLoader)
            for idx in docs:
                for key, value in idx.items():
                    if key == mode:
                        for k1, v1, in value.items():
                            cmd = 'self.' + k1 + '=' + repr(v1)
                            exec(cmd)
        f.close()
