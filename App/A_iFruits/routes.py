from . import app
from flask import render_template, Response , request , redirect, url_for , flash
import cv2 
import os
from .forms import uploadFile


execution_path = os.getcwd()
base_app = execution_path + '/App/A_iFruits/'
base_model = base_app + 'static/yolov5/'

# Page d'accueil (Description du produit)
@app.route('/')
def homepage():
    return render_template("home.html")



# -------------  Page upload d'une image pour detection

@app.route('/upload-photo', methods=['POST','GET'])
def upload_photo():


    form = uploadFile()
    description=[]
    if form.validate_on_submit():

        if form.file.data :
            photo = form.file.data
            # f = secure_filename(photo.filename) # Dont need it
            photo.save("App/A_iFruits/static/images/src/upload/file_upload.jpg")
            os.system(f'python {base_model}detect.py --source {base_app}static/images/src/upload/file_upload.jpg --weights {base_model}best4.pt --conf 0.5 --name yolo_foodex')
            
            # Run here the commande cli for yolo
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
            os.system(f'python {base_model}detect.py --source {base_app}static/images/dest/file_upload.jpg --weights {base_model}best4.pt --conf 0.5 --name yolo_foodex')
            flash('Video predicted', category='success')
            redirect(url_for('home.html')) # Page pour visualiser la vidéo upload et predite à faire (voir page live.html)
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

    flash('Suceess', category='success')
    return render_template("live.html")



@app.route('/video_feed')
def video_live():   
    os.system(f'python {base_model}detect.py --source 0 --weights {base_model}best4.pt --conf 0.5')
    
    video = cv2.VideoCapture(f'{base_model}runs/detect/exp/0.mp4') 

    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')