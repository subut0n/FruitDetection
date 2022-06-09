from A_iFruits import app
from flask import render_template, Response , request , redirect, url_for , flash
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from .forms import uploadFile
from werkzeug.utils import secure_filename


list_labels = [
    "Apple", "Apricot","Avocado","Banana", "Beetroot","Blueberry","Cabbage",
    "Cactus","Cantaloupe","Caspicum","Carambula","Carrot","Cauliflower","Cherry",
    "Chestnut","Clementine","Cocos","Corn","Cucumber","Dates","Eggplant","Fig",
    "Garlic","Ginger","Granadilla","Grape","Grapefruit","Guava","Hazelnut",
    "Huckleberry","Jalapeno","Kaki","Kiwi","Kohlrabi","Kumquats","Lemon","Limes",
    "Lychee","Mandarine","Mango","Mangostan","Maracuja","Melon","Mulberry",
    "Nectarine","Nut","Onion","Orange","Papaya","Passion","Peach","Pear",
    "Peas","Pepino","Pepper","Physalis","Pineapple","Pitahaya","Plum","Pomegranate",
    "Pomelo","Potato","Quince","Quince","Raddish","Rambutan","Raspberry","Redcurrant",
    "Salak","Soybeans","Spinach","Strawberry","Tamarillo","Tangelo","Tomato","Turnip",
    "Walnut","Watermelon"
]

# Page d'accueil (Description du produit)
@app.route('/')
def homepage():
    return render_template("home.html")



# -------------  Page upload d'une image pour detection

@app.route('/upload-photo', methods=['POST','GET'])
def upload_photo():


    form = uploadFile()
    execution_path = os.getcwd()
    description=[]
    if form.validate_on_submit():

        if form.file.data :
            photo = form.file.data
            # f = secure_filename(photo.filename) # Dont need it
            photo.save("App/A_iFruits/static/images/src/upload/file_upload.jpg")

            model = tf.keras.models.load_model("Data/aifruits_model.h5")
            predicting_img = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                dtype=tf.float32
            )
            img_transform = predicting_img.flow_from_directory(
                directory="App/A_iFruits/static/images/src",
                target_size=(75, 75)
            )
            description = list_labels[np.argmax(model.predict(img_transform))]
            # print(prediction)
            # detections = detector.detectObjectsFromImage(input_image=os.path.join(
            #     execution_path , "A_iFruits/static/images/src",  'file_upload.jpg'
            # ), output_image_path=os.path.join(execution_path , "A_iFruits/static/images/dest/predict_upload_file.jpg"))


            
        #     for eachObject in detections:
        #         text = eachObject["name"] , " : " , eachObject["percentage_probability"] 
        #         description.append(text)
        #     flash('File predicted', category='success')
        # else:
        #     flash('Please load an image', category="error" )


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
            photo.save(os.path.join(
                execution_path , "A_iFruits/static/images/dest",  'file_upload.avi'
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
    while True:
        success, image = video.read()

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed2')
def video_prediction():   

    execution_path = os.getcwd()
    video_path = os.path.join(execution_path, "A_iFruits/static/images/dest/file_upload.avi")
    video = cv2.VideoCapture(video_path) 

    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# ------------- Page detection d'objet en temps réel (webcam)



@app.route('/live')
def predict_live():

    video = cv2.VideoCapture(0) 
    execution_path = os.getcwd()
    # detector = VideoObjectDetection()
    # detector.setModelTypeAsRetinaNet()
    # detector.setModelPath( os.path.join(execution_path , "A_iFruits/static/models_files/resnet50_coco_best_v2.1.0.h5"))
    # detector.loadModel()

    # video_path = detector.detectObjectsFromVideo(camera_input=video,
    #                                 output_file_path=os.path.join(execution_path, "A_iFruits/static/images/dest/camera_detected_1")
    #                                 , frames_per_second=29, log_progress=True, minimum_percentage_probability=40)
    flash('Suceess', category='success')
    return render_template("live.html")



@app.route('/video_feed')
def video_live():   

    execution_path = os.getcwd()
    video = cv2.VideoCapture(0)
    video2 = cv2.VideoCapture(os.path.join(execution_path, "A_iFruits/static/images/dest/camera_detected_1.avi"))

    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')