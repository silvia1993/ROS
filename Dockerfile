FROM pytorch/pytorch
MAINTAINER Mohammad Reza Loghmani (loghmani@acin.tuwien.ac.at)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y --upgrade vim git
WORKDIR /relative-rotation
COPY . .
RUN pip install --upgrade pip
RUN pip install --upgrade -r requirements_ROS.txt
