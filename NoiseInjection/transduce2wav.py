"""https://www.python2.net/questions-890582.htm"""
"""https://kaen2891.tistory.com/49"""

import os
from pydub import AudioSegment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--files", type=str, required=True, help="변환하고 싶은 파일 디렉토리")
parser.add_argument("--output", type=str, required=True, help="저장하고 싶은 파일 디렉토리")

args = parser.parse_args()

list = os.listdir(args.files)

for name in list:
    folder = os.path.join(args.files, name)
    output = os.path.join(args.output, name)

    if name.endswith('flac'):
        output = output.replace('flac', 'wav')
        sound = AudioSegment.from_file(folder)
        sound.export(output, format="wav")
    elif name.endswith('mp3'):
        output = output.replace('mp3', 'wav')
        sound = AudioSegment.from_mp3(folder)
        sound.export(output, format="wav")
    else:
        pass




