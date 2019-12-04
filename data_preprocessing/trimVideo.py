

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
from os import listdir
from os.path import isfile, join
import os

mypath =  "./trimMovies"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("audio extraciont")
for i in onlyfiles:
    video = VideoFileClip("./trimMovies/" + i)
    audio = video.audio # 3.
    period = i.find('.')
    location = i[:period]+".wav"
    audio.write_audiofile(location) # 4.
    os.rename("./" + location, "./audioTrim/" + location)
    end_time = int(video.duration)
    print(end_time)
    ffmpeg_extract_subclip("./retrim/" + i, 0, end_time-12, targetname="./trimMovies/" + i)
    os.rename("./done_movies/" + i, "./temp/" + i)
    print("File Removed!")