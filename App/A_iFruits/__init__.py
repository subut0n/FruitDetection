from flask import Flask
import os
from dotenv import load_dotenv
load_dotenv() 

app = Flask(__name__)

from .routes import *

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['WTF_CSRF_SECRET_KEY'] = os.getenv('WTF_CSRF_SECRET_KEY')