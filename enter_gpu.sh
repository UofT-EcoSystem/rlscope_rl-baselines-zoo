#!/bin/bash
# Enter the docker gpu container

docker run -it --runtime=nvidia --rm --network host --ipc=host \
  --mount src=$(pwd),target=/root/code/stable-baselines,type=bind \
  "$@" rl-baselines-zoo
