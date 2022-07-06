# start with miniconda image
FROM continuumio/miniconda3

# setting the working directory 
WORKDIR /app

# Copy the file from your host to your current location in container
COPY . ./app

# Run the command inside your image filesystem to create an environment and name it in the requirements.yml file, in this case "myenv"
RUN apt-get update -y
RUN apt-get install libusb-1.0-0-dev -y 
RUN apt-get install ffmpeg libsm6 libxext6 v4l-utils -y
RUN conda env create --file ./app/conda-yolo.yml

# Activate the environment named "myenv" with shell command
SHELL ["conda", "run", "-n", "fruits2", "/bin/bash", "-c"]

# Make sure the environment is activated by testing if you can import flask or any other package you have in your requirements.yml file
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# exposing port 8050 for interaction with local host
EXPOSE 80

#Run your application in the new "myenv" environment

CMD ["conda", "run", "-n", "fruits2", "python", "./app/app.py"]