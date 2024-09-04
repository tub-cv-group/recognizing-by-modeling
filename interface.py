import argparse
import subprocess
import os

parser = argparse.ArgumentParser()

parser.add_argument('--image',
                    dest='image',
                    help='image_path',
                    default=None)

parser.add_argument('--audio',
                    dest='audio',
                    help='audio_path',
                    default=None)


args = parser.parse_args()
img=args.image
audio=args.audio


command = f'python3 run.py --config=configs/ia_attention_class.yaml --command=onecall --image={img} --audio={audio}'
subprocess.run(command, shell=True)
