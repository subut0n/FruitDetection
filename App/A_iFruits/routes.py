from A_iFruits import app
from flask import render_template, Response , request , redirect, url_for , flash
from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
import os
import cv2
from .forms import uploadFile
from werkzeug.utils import secure_filename
import matplotlib as plt

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

            photo = form.file.data
            # f = secure_filename(photo.filename) # Dont need it
            photo.save(os.path.join(
                execution_path , "A_iFruits/static/images/src",  'file_upload.jpg'
            ))

            
            
            detector = ObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath( os.path.join(execution_path , "A_iFruits/static/models_files/resnet50_coco_best_v2.1.0.h5"))
            detector.loadModel()

            detections = detector.detectObjectsFromImage(input_image=os.path.join(
                execution_path , "A_iFruits/static/images/src",  'file_upload.jpg'
            ), output_image_path=os.path.join(execution_path , "A_iFruits/static/images/dest/predict_upload_file.jpg"))


            
            for eachObject in detections:
                text = eachObject["name"] , " : " , eachObject["percentage_probability"] 
                description.append(text)
            flash('File predicted', category='success')
        else:
            flash('Please load an image', category="error" )


    return render_template("upload_photo.html", form=form, description=description)

# -------------  Page upload d'une video pour detection

execution_path = os.getcwd()

color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow', 'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown', 'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}


resized = False

def forFrame(frame_number, output_array, output_count, returned_frame):

    plt.clf()

    this_colors = []
    labels = []
    sizes = []

    counter = 0

    for eachItem in output_count:
        counter += 1
        labels.append(eachItem + " = " + str(output_count[eachItem]))
        sizes.append(output_count[eachItem])
        this_colors.append(color_index[eachItem])

    global resized

    if (resized == False):
        manager = plt.get_current_fig_manager()
        manager.resize(width=1000, height=500)
        resized = True

    plt.subplot(1, 2, 1)
    plt.title("Frame : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")

    plt.subplot(1, 2, 2)
    plt.title("Analysis: " + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

    plt.pause(0.01)


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
            
            detector = VideoObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath( os.path.join(execution_path , "A_iFruits/static/models_files/resnet50_coco_best_v2.1.0.h5"))
            detector.loadModel()

            video_path = detector.detectObjectsFromVideo(camera_input=os.path.join(
                execution_path , "A_iFruits/static/images/dest",  'file_upload.avi'), output_file_path=os.path.join(execution_path, "A_iFruits/static/images/dest/camera_detected_1")
                                            , frames_per_second=29, per_frame_function=forFrame, log_progress=True, minimum_percentage_probability=40)

            
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

    # video = cv2.VideoCapture(0) 
    # execution_path = os.getcwd()
    # detector = VideoObjectDetection()
    # detector.setModelTypeAsRetinaNet()
    # detector.setModelPath( os.path.join(execution_path , "A_iFruits/static/models_files/resnet50_coco_best_v2.1.0.h5"))
    # detector.loadModel()

    # video_path = detector.detectObjectsFromVideo(camera_input=video,
    #                                 output_file_path=os.path.join(execution_path, "A_iFruits/static/images/dest/camera_detected_1")
    #                                 , frames_per_second=1, log_progress=True, minimum_percentage_probability=40)
    flash('Suceess', category='success')
    return render_template("live.html")



@app.route('/video_feed')
def video_live():   

    execution_path = os.getcwd()
    video = cv2.VideoCapture(0)
    video2 = cv2.VideoCapture(os.path.join(execution_path, "A_iFruits/static/images/dest/camera_detected_1.avi"))

    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')