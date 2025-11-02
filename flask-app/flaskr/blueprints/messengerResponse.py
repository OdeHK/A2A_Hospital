import email
import functools
from hmac import new
import sys
import os

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('messenger', __name__, url_prefix='/messenger')

@bp.route('/')
def chat():
    data = {'domain' : request.host}
    return render_template('messenger/index.html' , data = data)