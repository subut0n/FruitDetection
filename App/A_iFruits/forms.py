
from flask_wtf.file import FileField
from flask_wtf import FlaskForm
from wtforms import SubmitField

class uploadFile(FlaskForm):
    file = FileField('file')
    submit = SubmitField(label="Submit file")