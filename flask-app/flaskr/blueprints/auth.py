import email
import functools
from hmac import new
import sys
import os
from sqlalchemy import select
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from .. import app , db , User , Role , Doctor


bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif not email:
          error = 'Email is required.'

        if error is None:
            try:
              new_user = User(username=username, email=email , password_hash=password)
              new_user.set_password()
              db.session.add(new_user)
              db.session.commit()
            except db.IntegrityError:
                error = f"User {username} is already registered."
            else:
                return redirect(url_for("auth.login"))

        flash(error)
    user_id = session.get('user_id')
    tempalte = 'index.html' if user_id is not None else 'auth/register.html'
    return render_template(tempalte)

@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None
        user = User(username=username)
        user = db.session.execute(db.select(User)).scalars().all()[0]

        if user is None:
            error = 'Incorrect username.'
        elif not user.check_password(password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user.id
            return redirect(url_for('homepage'))

        flash(error)
    user_id = session.get('user_id')
    tempalte = 'index.html' if user_id is not None else 'auth/login.html'
    return render_template(tempalte)

@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('homepage'))

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
      g.user = None
    else:
      g.user = session.get(User, user_id)