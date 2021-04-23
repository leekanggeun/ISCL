FROM tensorflow/tensorflow:2.5.0rc1-gpu

RUN apt-get update --fix-missing
RUN apt-get install -y vim
RUN apt-get install -y libtiff-dev
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get install -y tmux vim htop wget python-setuptools curl git

WORKDIR /home

COPY . .

RUN pip install --upgrade pip
RUN pip install -r /home/requirements.txt
