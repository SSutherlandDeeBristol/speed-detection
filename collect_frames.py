import cv2 as cv
import json
import os
import pickle as pkl
import skvideo.io

json_dir = '../info/train/'
vid_dir = '../videos/train/'
image_dir = '../images/train/'

if __name__ == '__main__':
    vid_map = {}
    json_map = {}
    image_map = {}

    image_counter = 0

    for i, filename in enumerate(os.listdir(json_dir)):
        if filename.endswith('.json'):
            json_map[filename[:-5]] = filename

    for i, filename in enumerate(os.listdir(vid_dir)):
        if filename.endswith('.mov'):
            vid_map[filename[:-4]] = filename

    for i, key in enumerate(vid_map.keys()):
        print(f'Processing {vid_map[key]}')
        if not key in json_map.keys():
            print('No JSON file found for video..')
            continue
        try:
            json_obj = json.load(open(os.path.join(json_dir, json_map[key])))

            if json_obj['gps'] is None or json_obj['startTime'] is None:
                print('No GPS or startTime data found..')
                continue

            print('Finding speeds and saving frames..')

            start_time = json_obj['startTime']

            vid = skvideo.io.VideoCapture(os.path.join(vid_dir, vid_map[key]))

            if int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)) != 1280 or int(vid.get(cv.CAP_PROP_FRAME_WIDTH) != 720):
                print('Video not high res..')
                continue

            for gps_entry in json_obj['gps']:
                timestamp = gps_entry['timestamp']
                speed = gps_entry['speed']

                vid.set(cv.CAP_PROP_POS_MSEC, int(timestamp - start_time))

                print(f'Scrubbed to time {timestamp - start_time}..')

                current_frame = vid.get(cv.CAP_PROP_POS_FRAMES)

                print(f'Current frame: {int(current_frame)}')

                print('Reading frame..')

                try:
                    success, current_image = vid.read()

                except:
                    print(f'Could not read frame {int(current_frame)}..')
                    continue

                if not success or current_frame < 1:
                    print(f'Could not read frame {int(current_frame)}..')
                    continue

                vid.set(cv.CAP_PROP_POS_FRAMES, current_frame - 1)

                print(f'Scrubbed to frame {int(current_frame - 1)}')

                print('Reading frame..')

                try:
                    success, prev_image = vid.read()
                except:
                    print(f'Could not read frame {int(current_frame - 1)}..')
                    continue

                if not success:
                    print(f'Could not read frame {current_frame - 1}..')
                    continue

                prev_filename = os.path.join(image_dir, f'{image_counter}-prev.jpg')
                current_filename = os.path.join(image_dir, f'{image_counter}-current.jpg')

                print(f'Writing image to {prev_filename}..')
                cv.imwrite(os.path.join(image_dir, f'{image_counter}-prev.jpg'), prev_image)
                print(f'Writing image to {current_filename}..')
                cv.imwrite(os.path.join(image_dir, f'{image_counter}-current.jpg'), current_image)

                image_map[image_counter] = (f'{image_counter}-prev.jpg', f'{image_counter}-current.jpg', speed)

                image_counter = image_counter + 1

            vid.release()

        except:
            print('Problem loading JSON..')

    pkl.dump(image_map, open('image_map.pkl', 'wb'))

    # cap = cv.VideoCapture('videos/train/00a0f008-a315437f.mov')

    # print(f'frame width: {cap.get(cv.CAP_PROP_FRAME_WIDTH)}')
    # print(f'frame height: {cap.get(cv.CAP_PROP_FRAME_HEIGHT)}')
    # print(f'fps: {cap.get(cv.CAP_PROP_FPS)}')
    # print(f'num frames: {cap.get(cv.CAP_PROP_FRAME_COUNT)}')