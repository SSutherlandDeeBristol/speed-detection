import cv2 as cv
import json
import os
import pickle as pkl
import ffmpeg

json_dir = '../info/train/'
vid_dir = '../videos/train/'
image_dir = '../images/train/'

def get_rotation_correct(path):
    meta_dict = ffmpeg.probe(path)

    rotation = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotation = cv.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotation = cv.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotation = cv.ROTATE_90_COUNTERCLOCKWISE

    return rotation

if __name__ == '__main__':
    vid_map = {}
    json_map = {}
    image_map = {}

    image_counter = 0

    for i, filename in enumerate(os.listdir(json_dir)):
        if filename.endswith('.json'):
            json_map[filename[:-5]] = os.path.join(json_dir, filename)

    for i, filename in enumerate(os.listdir(vid_dir)):
        if filename.endswith('.mov'):
            vid_map[filename[:-4]] = os.path.join(vid_dir, filename)

    for i, key in enumerate(vid_map.keys()):
        print(f'Processing {key} | {i+1}/{len(vid_map.keys())}')
        if not key in json_map.keys():
            print('No JSON file found for video..')
            continue
        try:
            json_obj = json.load(open(json_map[key]))

            if json_obj['gps'] is None or json_obj['startTime'] is None:
                print('No GPS or startTime data found..')
                continue

            # print('Finding speeds and saving frames..')

            start_time = json_obj['startTime']

            vid = cv.VideoCapture(vid_map[key])

            rotation = get_rotation_correct(vid_map[key])

            if int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)) != 1280 or int(vid.get(cv.CAP_PROP_FRAME_WIDTH) != 720):
                print('Video not high res..')
                continue

            for gps_entry in json_obj['gps']:
                timestamp = gps_entry['timestamp']
                speed = gps_entry['speed']

                if speed < 0:
                    print('Speed is negative..')
                    continue

                vid.set(cv.CAP_PROP_POS_MSEC, int(timestamp - start_time))

                # print(f'Scrubbed to time {timestamp - start_time}..')

                current_frame = vid.get(cv.CAP_PROP_POS_FRAMES)

                # print(f'Current frame: {int(current_frame)}')

                # print('Reading frame..')

                try:
                    success, current_image = vid.read()
                    if rotation is not None:
                        current_image = cv.rotate(current_image, rotation)
                except:
                    # print(f'Could not read frame {int(current_frame)}..')
                    continue

                if not success or current_frame < 1:
                    # print(f'Could not read frame {int(current_frame)}..')
                    continue

                vid.set(cv.CAP_PROP_POS_FRAMES, current_frame - 1)

                # print(f'Scrubbed to frame {int(current_frame - 1)}')

                # print('Reading frame..')

                try:
                    success, prev_image = vid.read()
                    prev_image = cv.rotate(prev_image, rotation)
                except:
                    # print(f'Could not read frame {int(current_frame - 1)}..')
                    continue

                if not success:
                    # print(f'Could not read frame {current_frame - 1}..')
                    continue

                prev_filename = os.path.join(image_dir, f'{image_counter}-prev.jpg')
                current_filename = os.path.join(image_dir, f'{image_counter}-current.jpg')

                # print(f'Writing image to {prev_filename}..')
                cv.imwrite(os.path.join(image_dir, f'{image_counter}-prev.jpg'), prev_image)
                # print(f'Writing image to {current_filename}..')
                cv.imwrite(os.path.join(image_dir, f'{image_counter}-current.jpg'), current_image)

                image_map[image_counter] = (f'{image_counter}-prev.jpg', f'{image_counter}-current.jpg', speed)

                image_counter = image_counter + 1

            vid.release()

        except:
            print('Problem loading JSON..')

    pkl.dump(image_map, open('image_map.pkl', 'wb'))