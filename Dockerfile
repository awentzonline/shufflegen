FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ADD . /app
WORKDIR /app
RUN pip install -e .
