FROM ubuntu:18.04

RUN apt-get update -y
#RUN apt-get upgrade python3 -y
RUN apt install python3-pip -y

RUN python3 --version

RUN pip --version

RUN pip install nltk

RUN pip freeze


