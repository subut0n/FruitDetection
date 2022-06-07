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
    return render_template("home.html")


