# FruitDetection
Computer Vision | Agile | Object Detection

You need python 3.7.6 and the right dependencies to run this app. 
Please foloow this following instructions : 

Follow this command to build the Dockerfile and run the App : 

Command : git https://github.com/subut0n/FruitDetection
Command : cd FruitDetection 
Command : docker build --tag foodex .
Command : docker run -d --network=host --device=/dev/video0:/dev/video0 foodex

Wait few seconds and access to the App in your localhost 
http://127.0.0.1:5000/
