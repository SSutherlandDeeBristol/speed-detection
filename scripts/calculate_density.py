import pickle as pkl
import argparse
from PIL import Image
import os
import numpy as np
import time
import sys
from scipy import stats
import math

parser = argparse.ArgumentParser(
    description="Calculate the optical flow density.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--mode',
                    default='train',
                    help='train/val/test',
                    required=True)

def calculate_mean_density(density_dict):
    mean_density_dict = dict()

    for k,v in density_dict.items():
        mean_density_dict[k] = np.mean(np.array(v))

    return mean_density_dict

if __name__ == '__main__':

    args = parser.parse_args()

    mode = args.mode

    optical_flow_images_path = f'../../{mode}/'

    run_name = 'bs_64_lr_0.001_run_85'
    logits_path = f'../logs/{run_name}/logits/14.pkl'

    # Load the testing data, scene labels and dataset map
    logits = pkl.load(open(logits_path, 'rb'))
    scene_labels = pkl.load(open(f'../../labels/labels_map_{mode}.pkl', 'rb'))
    dataset = pkl.load(open(f'../../{mode}/dataset_{mode}.pkl', 'rb'))

    dataset_length = len(dataset.keys())

    scene_density = dict()
    weather_density = dict()
    time_density = dict()

    density_error = list()
    density_speed = list()

    dataset_density = dict()

    start_time = time.time()

    for i, (filename, speed) in enumerate(dataset.values()):
        key = filename[:17]

        if key in scene_labels.keys():
            (weather, scene, time_of_day) = scene_labels[key]
        else:
            weather = scene = time_of_day = 'undefined'

        image = Image.open(os.path.join(optical_flow_images_path, f'{key}/{filename}'))

        image = np.array(image)

        num_non_zero = image.any(axis=-1).sum()

        density = num_non_zero / (image.shape[0] * image.shape[1])

        dataset_density[filename] = density

        if mode == 'val':
            pred, label = logits[filename]
            l1_error = abs(pred - label)
            density_error.append((density, l1_error))

        density_speed.append((speed, density))

        if weather == 'foggy' or weather == 'snowy':
            weather = 'other'

        weather_density.setdefault(weather, []).append(density)

        if scene == 'gas stations' or scene == 'parking lot' or scene == 'tunnel':
            scene = 'other'

        scene_density.setdefault(scene, []).append(density)

        time_density.setdefault(time_of_day, []).append(density)

        current_time = time.time()

        time_elapsed = current_time - start_time

        average_time = time_elapsed / (i+1)

        time_remaining = average_time * (dataset_length - (i+1))

        time_end = time.ctime(current_time + time_remaining)

        print(f'({i+1}/{dataset_length}): {filename} | time elapsed: {time_elapsed:.0f}s | time end: {time_end}')

    if mode == 'test':
        np.savetxt(f'plots_{mode}/density_error.csv',
            density_error, delimiter=',', fmt='%s')

    np.savetxt(f'plots_{mode}/density_speed.csv',
            density_speed, delimiter=',', fmt='%s')

    bin_means, bin_edges, binnumber = stats.binned_statistic(np.array(density_speed)[...,0],
                np.array(density_speed)[...,1], statistic='mean', bins=90, range=(0,45))

    np.savetxt(f'plots_{mode}/density_speed_hist.csv',
            [(float(edge), float(val) if not math.isnan(val) else 0.0) for edge, val in zip(bin_edges[1:], bin_means)], delimiter=',', fmt='%s')

    weather_mean_density = calculate_mean_density(weather_density)
    scene_mean_density = calculate_mean_density(scene_density)
    time_mean_density = calculate_mean_density(time_density)

    titles = [['x', 'label', 'value']]

    np.savetxt(os.path.join(f'plots_{mode}', f'{mode}_scene_density.csv'),
        titles + [(float(i+1), k, float(v)) for i,(k,v) in enumerate(scene_mean_density.items())],
        delimiter=',', fmt='%s')

    np.savetxt(os.path.join(f'plots_{mode}', f'{mode}_weather_density.csv'),
        titles + [(float(i+1), k, float(v)) for i,(k,v) in enumerate(weather_mean_density.items())],
        delimiter=',', fmt='%s')

    np.savetxt(os.path.join(f'plots_{mode}', f'{mode}_time_density.csv'),
        titles + [(float(i+1), k, float(v)) for i,(k,v) in enumerate(time_mean_density.items())],
        delimiter=',', fmt='%s')

    pkl.dump(dataset_density, open(f'../../{mode}/density_{mode}.pkl', 'wb'))
