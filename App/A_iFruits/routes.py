from App.A_iFruits import app
from flask import render_template, Response , request , redirect, url_for , flash
import numpy as np
import os
from .forms import uploadFile
from werkzeug.utils import secure_filename

import pandas as pd
from .prediction import visualize, ObjectDetectorOptions, ObjectDetector, return_class_names
from PIL import Image
import cv2



list_funfact = pd.read_csv('Data/fruits_description.csv')

# Page d'accueil (Description du produit)
@app.route('/')
def homepage():
    return render_template("home.html")



# -------------  Page upload d'une image pour detection

@app.route('/upload-photo', methods=['POST','GET'])
def upload_photo():


    form = uploadFile()
    description=[]
    funfact_text=[]
    sent_text = []
    unique_class=[]
    counter = []

    TFLITE_MODEL_PATH = "App/A_iFruits/static/models_files/foodex-v6-lite0.tflite" #@param {type:"string"}
    IMAGES_FOLDER = 'App/A_iFruits/static/images/src/upload/'
    FILE_NAME = 'file_upload.jpg'
    DETECTION_THRESHOLD = 0.55 #@param of confidence before display the prediction


    if form.validate_on_submit():

        if form.file.data :

            photo = form.file.data
            photo.save(IMAGES_FOLDER + FILE_NAME)

            image = Image.open(IMAGES_FOLDER + FILE_NAME).convert('RGB')
            image.thumbnail((500, 500), Image.ANTIALIAS)
            image_np = np.asarray(image)


            # Load the TFLite model
            options = ObjectDetectorOptions( num_threads=4, score_threshold=DETECTION_THRESHOLD)

            detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)

            # Run object detection .
            detections = detector.detect(image_np)

            image_np = visualize(image_np, detections)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(IMAGES_FOLDER , FILE_NAME),  image_np)

            description = return_class_names(image_np, detections)

            for class_name in description:
                name_only = class_name.split(" ", 1)[0]
                if name_only not in unique_class:
                    unique_class.append(name_only)
                    funfact_text = funfact_text + list_funfact['description'].loc[list_funfact['name']==name_only].values.tolist()

            counter = range(len(unique_class))

    return render_template("upload_photo.html", form=form, description=description, funfact_text=funfact_text, unique_class=unique_class, counter=counter)

# -------------  Page upload d'une video pour detection

@app.route('/upload-video', methods=['POST','GET'])
def upload_video():

    form = uploadFile()
    execution_path = os.getcwd()
    description = []
    if form.validate_on_submit():

        if form.file.data :


            print(form.file.data)
            photo = form.file.data
            photo.save(os.path.join(
                execution_path , "App/A_iFruits/static/images/dest",  'file_upload.avi'
            ))

            
            print(form.file.__dict__)
            
            # detector = VideoObjectDetection()
            # detector.setModelTypeAsRetinaNet()
            # detector.setModelPath( os.path.join(execution_path , "A_iFruits/static/models_files/resnet50_coco_best_v2.1.0.h5"))
            # detector.loadModel()

            # video_path = detector.detectObjectsFromVideo(camera_input=os.path.join(
            #     execution_path , "A_iFruits/static/images/dest",  'file_upload.avi'), output_file_path=os.path.join(execution_path, "A_iFruits/static/images/dest/camera_detected_1")
            #                                 , frames_per_second=1, log_progress=True, minimum_percentage_probability=40)

            
            flash('File predicted', category='success')
        else:
            flash('Please load an image', category="error" )

    return render_template("upload_video.html", form=form, description=description)
    

# Fonction generateur de video 
def gen(video):


    TFLITE_MODEL_PATH = "App/A_iFruits/static/models_files/foodex-v6-lite0.tflite" #@param {type:"string"}
    DETECTION_THRESHOLD = 0.7 #@param of confidence before display the prediction
    options = ObjectDetectorOptions( num_threads=4, score_threshold=DETECTION_THRESHOLD)
    detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)


    while True:
        success, image = video.read()
        # Run object detection . (too long)
        detections = detector.detect(image)
        image = visualize(image, detections)
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed2')
def video_prediction():   

    execution_path = os.getcwd()
    video_path = os.path.join(execution_path, "App/A_iFruits/static/images/dest/file_upload.avi")
    video = cv2.VideoCapture(video_path) 

    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# ------------- Page detection d'objet en temps réel (webcam)



@app.route('/live')
def predict_live():

    return render_template("live.html")



@app.route('/video_feed')
def video_live():   

    execution_path = os.getcwd()
    video = cv2.VideoCapture(0)

    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')