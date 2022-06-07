from A_iFruits import app
from flask import render_template

# Page d'accueil (Description du produit)
@app.route('/')
def homepage():
    return render_template("home.html")

# Page upload d'une image pour detection
@app.route('/upload')
def predict_upload():
    return render_template("home.html")

# Page detection d'objet en temps réel (webcam)
@app.route('/live')
def predict_live():
    from imageai.Detection import ObjectDetection
    import os

    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "A_iFruits/static/models_files/resnet50_coco_best_v2.1.0.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "A_iFruits/static/images/src/image2.jpg"), output_image_path=os.path.join(execution_path , "A_iFruits/static/images/dest/imagenew2.jpg"))

    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

    return render_template("home.html")


