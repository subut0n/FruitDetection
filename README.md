Computer Vision | Agile | Object Detection

# FruitDetection

Welcome AI learner, 

Here you can take a picture about fruits and our model will predict it between 65 categories ! 

Link websites access : 
  - EfficientNet0 model with Tensorflow : https://foodexv0.azurewebsites.net/
  - Yolo model with Pytorch : https://foodexv1.azurewebsites.net/upload-photo
  
  
----------------------------

For now if you want to use webcam real time mode, you need to deploy it in local  

Please follow this following instructions : 

Command : git clone https://github.com/subut0n/FruitDetection

Command : cd FruitDetection 

Command : docker build --tag foodex .

Command : docker run -d --network=host --device=/dev/video0:/dev/video0 foodex


Wait few seconds and access to the App at your localhost port 8080
or at this link http://localhost:8080/ 


----------------------------
Students by Simplon 
