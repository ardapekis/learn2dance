from pytube import YouTube


import sys
from moviepy.editor import *

from os import listdir
from os.path import isfile, join



playlistLinks = []
i = 0
for v in playlistLinks:
    filenam = "temp" + str(i)
    YouTube(v).streams.first().download(output_path="./movies", filename=filenam)
mypath =  "./movies"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("audio extraciont")
for i in onlyfiles:
    video = VideoFileClip("./movies/" + i) # 2.
    audio = video.audio # 3.
    location = i[:len(i)-4]+".wav"
    audio.write_audiofile(location) # 4.
    os.rename("./" + location, "./audio/" + location)