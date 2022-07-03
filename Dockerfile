FROM continuumio/miniconda3

WORKDIR /app
COPY . ./app

RUN apt-get update -y
RUN apt-get install libusb-1.0-0-dev -y 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN conda env create -f ./app/conda.yml

# Make RUN commands use the new environment:
RUN echo "conda activate fruits" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# The code to run when container is started:
RUN chmod +x ./app/entrypoint.sh
ENTRYPOINT ["./app/entrypoint.sh"]