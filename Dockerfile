FROM python:3.8-slim-buster
WORKDIR /

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

# CMD [ "python3"]
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install qt5-default -y
COPY main.py main.py
# RUN python3 main.py 