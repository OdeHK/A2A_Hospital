import os

from flask import Flask
from .models import db , User , Role , Doctor
UPLOAD_FOLDER = "flaskr/static/uploads"

app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY='dev',
    DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
)

# a simple page that says hello
app.config["SQLALCHEMY_DATABASE_URI"] =  "mysql+pymysql://root:password@localhost/hopitalagent"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]  =  False
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

db.init_app(app=app)

if __name__ == '__main__':
  with app.app_context():
    db.create_all()
    print("Database tables created!")