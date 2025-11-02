import os
from flask import Flask , render_template
# from .models import db , User , Role;
from .config import app ,  db , User , Role , Doctor


def create_app(test_config=None):

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def homepage():
        return render_template('index.html')
    @app.route('/index')
    def index():
        return render_template('index.html')
    
    from .blueprints import auth , messengerResponse , importData
    app.register_blueprint(auth.bp)
    app.register_blueprint(messengerResponse.bp)
    app.register_blueprint(importData.bp)
    

    return app

if __name__ == 'flaskr':
    app = create_app()