from A_iFruits import app
from flask import render_template, Response , request , redirect, url_for , flash
from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
import os
import cv2
from .forms import uploadFile
from werkzeug.utils import secure_filename

# Page d'accueil (Description du produit)
@app.route('/')
def homepage():
    return render_template("home.html")



# -------------  Page upload d'une image pour detection

@app.route('/upload-photo', methods=['POST','GET'])
def upload_photo():


    form = uploadFile()
    execution_path = os.getcwd()
    description = []
    if form.validate_on_submit():

        if form.file.data :

            print(form.file.data)
            photo = form.file.data
            f = secure_filename(photo.filename)
            photo.save(os.path.join(
                execution_path , "A_iFruits/static/images/dest",  f
            ))

            
            print(form.file.__dict__)
            
            detector = ObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath( os.path.join(execution_path , "A_iFruits/static/models_files/resnet50_coco_best_v2.1.0.h5"))
            detector.loadModel()

            detections = detector.detectObjectsFromImage(input_image=os.path.join(
                execution_path , "A_iFruits/static/images/src",  f
            ), output_image_path=os.path.join(execution_path , "A_iFruits/static/images/dest/predict_upload_photo.jpg"))


            
            for eachObject in detections:
                text = eachObject["name"] , " : " , eachObject["percentage_probability"] 
                description.append(text)
            flash('File predicted', category='success')
        else:
            flash('Please load an image', category="error" )


    return render_template("upload_photo.html", form=form, description=description)

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
            f = secure_filename(photo.filename)
            print(f)
            photo.save(os.path.join(
                execution_path , "A_iFruits/static/images/dest",  f
            ))

            
            print(form.file.__dict__)
            
            detector = ObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath( os.path.join(execution_path , "A_iFruits/static/models_files/resnet50_coco_best_v2.1.0.h5"))
            detector.loadModel()

            video_path = detector.detectObjectsFromVideo(camera_input=os.path.join(
                execution_path , "A_iFruits/static/images/src",  f), output_file_path=os.path.join(execution_path, "A_iFruits/static/camera_detected_1")
                                            , frames_per_second=29, log_progress=True, minimum_percentage_probability=40)

            
            print(video_path)
            flash('File predicted', category='success')
        else:
            flash('Please load an image', category="error" )

    return render_template("upload_video.html", form=form, description=description)



# ------------- Page detection d'objet en temps réel (webcam)

# Fonction generateur de video 
def gen(video):
    while True:
        success, image = video.read()

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():   
    video = cv2.VideoCapture(0) 

    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def predict_live():
    video = cv2.VideoCapture(0) 

        
    
    # execution_path = os.getcwd()
    # detector = VideoObjectDetection()
    # detector.setModelTypeAsRetinaNet()
    # detector.setModelPath( os.path.join(execution_path , "A_iFruits/static/models_files/resnet50_coco_best_v2.1.0.h5"))
    # detector.loadModel()

    # # -- le video_path creer un fichier .avi (à voir comment le lire) --

    # video_path = detector.detectObjectsFromVideo(camera_input=video,
    #                                 output_file_path=os.path.join(execution_path, "A_iFruits/static/camera_detected_1")
    #                                 , frames_per_second=29, log_progress=True, minimum_percentage_probability=40)
    
    # print(video_path)
    flash('Suceess', category='success')
    return render_template("live.html", video=video)



