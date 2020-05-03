import pickle as pkl
from matplotlib import pyplot as plt
import numpy as np
import torch
import os

run_name = 'bs_64_lr_0.001_run_85'
file_name = f'../logs/{run_name}/logits/14.pkl'

def get_errors(results_map):
    errors = dict()
    loss_function = torch.nn.SmoothL1Loss()

    for k, results in results_map.items():
        preds = np.array(results)[...,0]
        labels = np.array(results)[...,1]

        loss = loss_function(torch.Tensor(labels), torch.Tensor(preds))
        root_mse = np.sqrt(np.mean([(p-l)**2 for p,l in zip(preds, labels)]))
        mean_l1 = np.mean(abs(preds-labels))
        errors[k] = [loss.item(), root_mse, mean_l1]

        # print(f'{k}:\nloss: {loss}\nRMSE: {root_mse}\nmean l1 error: {mean_l1}')

    return errors

def get_largest_errors(logits, test_set, num):
    errors = [[f,p,l] for (p,l), (f,_) in zip(logits.values(), test_set.values())]

    errors = sorted(errors, key=lambda x: abs(x[1] - x[2]), reverse=True)

    print(errors[:num])

if __name__ == '__main__':
    logits = pkl.load(open(file_name, 'rb'))
    scene_labels = pkl.load(open('../../labels/labels_map_val.pkl', 'rb'))
    test_set = pkl.load(open('../../val/dataset_val.pkl', 'rb'))

    scene_results = dict()
    weather_results = dict()
    time_results = dict()

    for (pred, label), (filename, _) in zip(logits.values(), test_set.values()):
        (weather, scene, time) = scene_labels[filename[:17]]

        if filename == 'b1d7b3ac-36f2d3b7-8.png':
            print(f'pred: {pred}, label: {label}')

        if weather in ['foggy', 'snowy']:
            weather = 'other'

        if scene in ['gas stations', 'parking lot', 'tunnel']:
            scene = 'other'

        scene_results.setdefault(scene, []).append([pred, label])
        weather_results.setdefault(weather, []).append([pred, label])
        time_results.setdefault(time, []).append([pred, label])

    scene_errors = get_errors(scene_results)
    weather_errors = get_errors(weather_results)
    time_errors = get_errors(time_results)

    np.savetxt(os.path.join('performance_csv', f'scene_performance.csv'),
        [(float(i+1), k, float(v[0]), float(v[1]), float(v[2])) for i,(k,v) in enumerate(scene_errors.items())],
        delimiter=',', fmt='%s')

    np.savetxt(os.path.join('performance_csv', f'weather_performance.csv'),
        [(float(i+1), k, float(v[0]), float(v[1]), float(v[2])) for i,(k,v) in enumerate(weather_errors.items())],
        delimiter=',', fmt='%s')
    np.savetxt(os.path.join('performance_csv', f'time_performance.csv'),
        [(float(i+1), k, float(v[0]), float(v[1]), float(v[2])) for i,(k,v) in enumerate(time_errors.items())],
        delimiter=',', fmt='%s')