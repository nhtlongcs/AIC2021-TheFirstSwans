import os
import parser
import re
import subprocess
from argparse import ArgumentParser

import joblib
import optuna

SUBMISSION_FOLDER = ''


def blackbox(params):
    dict_thrs = params['dict_thrs']
    os.system(
        f'python ensemble/update_dict.py {SUBMISSION_FOLDER}/* --threshold {dict_thrs}')
    os.system(f'python convert/label2pred.py outputs/')
    out = subprocess.check_output(
        f'python evaluation_det_e2e_offline/main.py VIN text_results.json', shell=True).decode()

    template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
    pos = out.rfind('E2E_RESULTS')
    parse_res = re.match(template, out[pos:]).groups()
    print(parse_res)
    hmean = parse_res[-1]
    hmean = float(re.match(r'\d+\.\d+', hmean).group())
    return hmean


def objective(trial):
    x = trial.suggest_float('x', 0.1, 0.5)
    return blackbox({'dict_thrs': x})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--folder-submission', type=str, required=True)
    parser.add_argument('--study-name', type=str, default='base')
    args = parser.parse_args()
    SUBMISSION_FOLDER = args.folder_submission
    study = optuna.create_study(
        direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=100)

    print(study.best_params)  # E.g. {'x': 0.11369181941404821}}
    joblib.dump(study, f'{args.study_name}.pkl')
