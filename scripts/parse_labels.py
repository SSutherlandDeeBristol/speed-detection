import numpy as np
import pickle as pkl
import json
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Collect the labels.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--mode',
                    default='train',
                    help='train/val/test',
                    required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    mode = args.mode

    labels_dir = '../../labels/'
    vid_dir = f'../../{mode}/'
    dataset_dir = f'../../{mode}/dataset_{mode}.pkl'
    plot_dir = f'plots_{mode}/'

    weather_types = ['clear', 'rainy', 'undefined', 'foggy', 'snowy', 'overcast', 'partly cloudy']
    scene_types = ['residential', 'undefined', 'city street', 'highway', 'tunnel', 'gas stations', 'parking lot']
    time_of_day_types = ['daytime', 'dawn/dusk', 'undefined', 'night']

    # id -> (weather, scene, timeofday)
    labels_map = {}

    labels_data = json.load(open(os.path.join(labels_dir, f'bdd100k_labels_images_{mode}.json')))

    for i, label in enumerate(labels_data):

        label_id = label['name'][:-4]
        print(f'Processing {label_id} ({i+1}/{len(labels_data)})', flush=True)

        attributes = label['attributes']

        weather = attributes['weather']
        scene = attributes['scene']
        time_of_day = attributes['timeofday']

        labels_map[label_id] = (weather, scene, time_of_day)

    pkl.dump(labels_map, open(os.path.join(labels_dir, f'labels_map_{mode}.pkl'), 'wb'))

    dataset = pkl.load(open(dataset_dir, 'rb'))

    weather = []
    scene = []
    time_of_day = []

    for i, data in enumerate(dataset.values()):
        id = (data[0])[:17]
        if id in labels_map.keys():
            weather.append(labels_map[id][0])
            scene.append(labels_map[id][1])
            time_of_day.append(labels_map[id][2])
        else:
            weather.append('undefined')
            scene.append('undefined')
            time_of_day.append('undefined')

    weather_map = {}
    scene_map = {}
    time_of_day_map = {}

    for weather_type in weather_types:
        n = weather.count(weather_type)
        if weather_type == 'foggy' or weather_type == 'snowy':
            v = weather_map.setdefault('other', 0)
            weather_map['other'] = v + n
        else:
            weather_map[weather_type] = n

    for scene_type in scene_types:
        n = scene.count(scene_type)
        if scene_type == 'gas stations' or scene_type == 'parking lot' or scene_type == 'tunnel':
            v = scene_map.setdefault('other', 0)
            scene_map['other'] = v + n
        else:
            scene_map[scene_type] = n

    for time_of_day_type in time_of_day_types:
        n = time_of_day.count(time_of_day_type)
        time_of_day_map[time_of_day_type] = n

    try:
        os.mkdir(plot_dir)
    except FileExistsError:
        pass

    blue = (14/255, 107/255, 176/255)
    red = (197/255, 42/255, 20/255)

    color = blue if args.mode == 'train' else red

    plt.rc('font', size=32)

    plt.figure(figsize=(20,10))
    plt.bar(weather_map.keys(), weather_map.values(), color=color)
    plt.ylabel('Frequency', fontsize=34, labelpad=30)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{mode}_weather.pdf'), bbox_inches=None, transparent=True)

    plt.figure(figsize=(20,10))
    plt.bar(scene_map.keys(), scene_map.values(), color=color)
    plt.ylabel('Frequency', fontsize=34, labelpad=30)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{mode}_scene.pdf'), bbox_inches=None, transparent=True)

    plt.figure(figsize=(20,10))
    plt.bar(time_of_day_map.keys(), time_of_day_map.values(), color=color)
    plt.ylabel('Frequency', fontsize=34, labelpad=30)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{mode}_time_of_day.pdf'), bbox_inches=None, transparent=True)