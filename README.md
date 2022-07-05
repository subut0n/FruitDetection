# FruitDetection
Computer Vision | Agile | Object Detection

You have access to our computer vision models which could detect almost 65 fruits.

Please foloow this following instructions : 

Follow this command to build the Dockerfile and run the App in localhost: 

Command : git clone https://github.com/subut0n/FruitDetection

Command : cd FruitDetection 

Command : docker build --tag foodex .

Command : docker run -d --network=host --device=/dev/video0:/dev/video0 foodex


Wait few seconds and access to the App at this link http://127.0.0.1:8000/ 

