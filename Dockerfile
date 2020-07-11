FROM pytorch/pytorch
MAINTAINER Silvia Bucci & Mohammad Reza Loghmani
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y --upgrade vim git
WORKDIR /ROS
COPY . .
RUN pip install --upgrade pip
RUN pip install --upgrade -r requirements_ROS.txt
