#!/bin/zsh

# run docker container
sudo nvidia-docker run -ti \
	-h mwfa \
	--name mwfa \
	--ipc host \
	-v $PWD:/root/mwfa:rw \
	-v $HOME/Documents/VADSET:/root/VADSET:ro \
	$USER/cuda:latest
