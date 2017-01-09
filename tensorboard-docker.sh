# Run tensorboard docker in background and link ./logs to /root/logs
# Doesn't work if there is no logs to run. Home directory contains
# logs
docker run -d -p 6006:6006 \
	--name tensorboard \
	-v $(pwd):/home/jovyan/work \
	jupyter/tensorflow-notebook \
	tensorboard --logdir ./logs