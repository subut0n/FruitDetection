FROM continuumio/miniconda3

WORKDIR /app
COPY . ./app

ENV PORT 5000
EXPOSE 5000

RUN apt-get update -y
RUN apt-get install libusb-1.0-0-dev -y 
RUN apt-get install ffmpeg libsm6 libxext6 v4l-utils -y
RUN conda env create -f ./app/conda-yolo.yml
RUN chmod +x ./app/entrypoint.sh
ENTRYPOINT ["./app/entrypoint.sh"]