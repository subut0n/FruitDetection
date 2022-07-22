# start with miniconda image
FROM continuumio/miniconda3

# setting the working directory 
WORKDIR /app

# Copy the file from your host to your current location in container
COPY . ./app

# Run the command inside your image filesystem to create an environment and name it in the requirements.yml file, in this case "myenv"
RUN apt-get update -y
RUN apt-get install libusb-1.0-0-dev -y 
RUN apt-get install ffmpeg libsm6 libxext6 v4l-utils '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev -y
RUN conda env create --file ./app/conda.yml

# Activate the environment named "myenv" with shell command
SHELL ["conda", "run", "-n", "fruits", "/bin/bash", "-c"]

# Make sure the environment is activated by testing if you can import flask or any other package you have in your requirements.yml file
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# exposing port 8050 for interaction with local host
EXPOSE 8080

#Run your application in the new "myenv" environment
RUN chmod +x ./app/entrypoint.sh
RUN chmod +x ./app/stream.sh
ENTRYPOINT ["./app/entrypoint.sh", "./app/stream.sh"]