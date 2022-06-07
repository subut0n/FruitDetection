from A_iFruits import app
from flask import render_template

@app.route('/')
def homepage():
    return render_template("home.html")