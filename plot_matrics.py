import json
import numpy as np
from irl_dcb import metrics


def print_scanpath_ratio(asset_name):
    with open(asset_name + 'predicted_scanpaths.json') as json_file:
        trajs = json.load(json_file)
    bbox_annos = np.load('data/bbox_annos.npy',allow_pickle=True).item()
    average_scanpath_ratio = 0
    for i in range(len(trajs)):
        name = trajs[i]['task'] + '_' + trajs[i]['name']
        scanpath_ratio = metrics.scanpath_ratio(trajs[i], bbox_annos[name])
        average_scanpath_ratio += scanpath_ratio

    average_scanpath_ratio = average_scanpath_ratio / len(trajs)
    print(average_scanpath_ratio)


def print_multimatch(asset_name):
    with open(asset_name + 'predicted_scanpaths.json') as json_file:
        pred_trajs = json.load(json_file)
    with open('data/human_scanpaths_TP_trainval_valid.json') as json_file:
        human_trajs = json.load(json_file)
    mm = metrics.compute_mm(human_trajs, pred_trajs, 320, 512)
    print(mm)


if __name__ == '__main__':
    assets = ['assets/log_20221130_1722/', 'assets/log_20221208_1725/', 'assets/log_20221208_2334/']
    for asset_name in assets:
        print_scanpath_ratio(asset_name)
        print_multimatch(asset_name)
