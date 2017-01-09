FROM jupyter/scipy-notebook

USER $NB_USER

# Install Python 3 Tensorflow
RUN conda install --quiet --yes -c jjhelmus 'tensorflow=0.12.0rc0'

ESPOSE 8888 6006
