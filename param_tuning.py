import pandas as pd
import optuna
import polars as pl
import sys
sys.setrecursionlimit(10**7) # 再起回数の設定
from pathlib import Path
import subprocess
from joblib import Parallel, delayed
from matplotlib import pyplot
import seaborn as sns
from tqdm.notebook import tqdm as tqdm
import numpy as np
import os


inbound = Path('./in')
outbound = Path('./out')


def test(command, inbound, i):
    result = subprocess.run(f"{command} < {inbound}/{i:04d}.txt", 
            shell=True, # コマンドを文字列で渡すオプション
            cwd=".", # カレントワーキングディレクトリ、コマンドを実行するディレクトリ
            capture_output=True, # 実行結果をresultに代入する
            text=True # 実行結果を文字列で渡す)
        )
        # proc = subprocess.Popen(f"{command} {inbound}/{i:04d}.txta {outbound}/{i:04d}.txt", shell=True, cwd='.')
    score = int(result.stdout.split()[-1]) # 大抵、標準出力の一番最後にスコアが出力される
    return score
        

def investigate(s_ss, e_ss, bw, md, thr,  l=100, inbound='in', outbound='out'):

    command = f"./test {s_ss[0]} {s_ss[1]} {e_ss[0]} {e_ss[1]} {bw[0]} {bw[1]} {md[0]} {md[1]} {thr}"
        # command = "cargo run --release --bin vis"
    # ↑いつものAHCのビジュアライザを使う場合
    res = []
    for i in tqdm(range(l)):
        res.append(test(command, inbound, i))
    return np.mean(res)

def objective(trial):
    s_ss_s = trial.suggest_int("s_ss_s", 1, 20)
    s_ss_e = trial.suggest_int("s_ss_e", 1, 20)
    e_ss_s = trial.suggest_int("e_ss_s", 1, 20)
    e_ss_e = trial.suggest_int("e_ss_e", 1, 20)
    bw_s = trial.suggest_int("bw_s", 1, 20)
    bw_e = trial.suggest_int("bw_e", 1, 20)
    md_s = trial.suggest_int("md_s", 1, 20)
    md_e = trial.suggest_int("md_e", 1, 20)
    thr = trial.suggest_int("thr", 5, 14)
    ret = investigate((s_ss_s, s_ss_e), (e_ss_s, e_ss_e), (bw_s, bw_e), (md_s, md_e), thr)
    return ret

def run(study_name: str):
    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///./f_study.db",
    )
    study.optimize(objective, n_trials=50)
    return os.getpid()


if __name__ == "__main__":
    study_name = "f_study"
    study = optuna.create_study(
    study_name=study_name,
    direction="minimize",
    storage="sqlite:///./f_study.db")
    # joblibでプロセス並列化
    process_ids = Parallel(n_jobs=4)([delayed(run)(study_name) for _ in range(6)])
    print(process_ids)  # [14937, 14938]
