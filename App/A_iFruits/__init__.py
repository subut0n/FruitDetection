from flask import Flask

app = Flask(__name__)

from .routes import *

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

WTF_CSRF_SECRET_KEY = 'a random string'
app.config['WTF_CSRF_SECRET_KEY'] = SECRET_KEY
