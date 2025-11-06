import email
import functools
from hmac import new
import sys
import os
from services.textToSpeechService import SpeedService
speed_service = SpeedService()
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for , send_file
)
bp = Blueprint('messenger', __name__, url_prefix='/messenger')

@bp.route('/')
def chat():
    data = {'domain' : request.host}
    return render_template('messenger/index.html' , data = data)

@bp.route('/voice-chat' , methods = ['POST'])
def createVoiceChat():
    text = request.get_json()
    text = text['text']
    if text:
        path_file = speed_service.convertTextToSpeed(text)
    if path_file:
        return {
            'status' : True ,
            'result' :  path_file,
        }
    return {
        'status' : False,
        'message' : 'Can not create file speed'
    }