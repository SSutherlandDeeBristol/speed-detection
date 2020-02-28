import pexpect
import os
import pickle as pkl
import argparse
import sys

parser = argparse.ArgumentParser(
    description="Collect the frames from video, generate the optical flow images and upload them to bc4.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode',
                    default='train',
                    help='train/val',
                    required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    mode = args.mode

    collect_frames = pexpect.spawn(f'python3 collect_frames.py --mode={mode}', encoding='utf-8', timeout=None)
    collect_frames.logfile = sys.stdout
    collect_frames.read()

    print('Collected Frames..')

    optical_flow = pexpect.spawn(f'python3 generate_optical_flow.py --mode={mode}', encoding='utf-8', timeout=None)
    optical_flow.logfile = sys.stdout
    optical_flow.read()

    print('Generated Optical Flow..')

    upload_output = pexpect.spawn(f'python3 upload_new_files.py --mode={mode}', encoding='utf-8', timeout=None)
    upload_output.logfile = sys.stdout
    upload_output.read()

    print('Uploaded files..')

    print('Finished.')
