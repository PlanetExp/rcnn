# Run tensorflow-notebook docker with exposed ports to jupyter and 
# and link to current folder pass:pass
# To get a new password use IPython.lib.passwd to generate a new one
# and replace sha1:...
docker run -d -p 8888:8888 \
	--name jupyter \
	-v $(pwd):/home/jovyan/work \
	jupyter/tensorflow-notebook \
	start-notebook.sh --NotebookApp.password='sha1:1c24bff04dea:dff805b7a96d82bcb546d4807a2e96a7f3789c2e'