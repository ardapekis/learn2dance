#!/bin/bash
for filename in videos/*; do
	./build/examples/openpose/openpose.bin --video videos/$filename --write_json output/$filename --display 0 --render_pose 0
done
