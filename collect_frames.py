import cv2 as cv
import json
import os
import pickle as pkl

json_dir = 'info/train/'
vid_dir = 'videos/train/'

vid_map = {}
vid_to_json = {}

# TODO: save the name of the JSONs that have the videos currently in the vid_dir directory
# TODO: pickle a dict from video filename -> JSON file for easy reading
if __name__ == '__main__':
    for filename in os.listdir(vid_dir):
        if not filename.endswith('.mov'):
            continue

        vid_map[filename] = os.path.join(vid_dir, filename)

    pkl.dump(vid_to_json, open('vid_to_json.pkl', 'w'))

    for i, filename in enumerate(os.listdir(json_dir)):
        if not filename.endswith('.json'):
            continue

        # print(f'reading {os.path.join(json_dir, filename)}')
        print(i)

        try:
            json_obj = json.load(open(os.path.join(json_dir, filename), 'r'))

            if json_obj['filename'] is None:
                continue

            if json_obj['filename'] in vid_map.keys():
                print("hooray")
                vid_to_json[json_obj['filename']] = filename
        except:
            continue

    

        # cap = cv.VideoCapture('videos/train/00a0f008-a315437f.mov')

        # print(f'frame width: {cap.get(cv.CAP_PROP_FRAME_WIDTH)}')
        # print(f'frame height: {cap.get(cv.CAP_PROP_FRAME_HEIGHT)}')
        # print(f'fps: {cap.get(cv.CAP_PROP_FPS)}')
        # print(f'num frames: {cap.get(cv.CAP_PROP_FRAME_COUNT)}')