import os
import datetime
from pathlib import Path



def init_logdirs(
    experiment_name, log_path
):
    ckpt_dir = GetCheckPointsPath(experiment_name, log_path)
    log_path = GetLogsPath(experiment_name, log_path)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        

def GetCheckPointsPath(
    experiment_name, log_path
):
    log_path = Path(log_path)
    return log_path / experiment_name / "CheckPoints/"


def GetLogsPath(
    experiment_name, log_path
):
    log_path = Path(log_path)
    return log_path / experiment_name / "Logs/"


def LogText(
    text, experiment_name, log_path
):
    
    Exp_log_dir = GetLogsPath(experiment_name, log_path)
    Log_file = Exp_log_dir / (experiment_name + '.txt')

    text_to_print = text + "(" + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + ")" + '\n'
    print(str(text_to_print))

    f = open(Log_file, 'a')
    f.write(text_to_print)
    f.close()
