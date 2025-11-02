import email
import functools
from hmac import new
import sys
import os
from sqlalchemy import select
import pandas as pd

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.utils import secure_filename

from .. import app , db , Doctor

UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('..\\static\\upload', cur_path)

bp = Blueprint('import_data', __name__, url_prefix='/config')

@bp.route('/' ,  methods = ['GET'])
def importData():
    return render_template('import_data/index.html')

@bp.route('/doctor' , methods=['POST']) 
def add_doctor():
    try:
        name = request.form.get('name')
        age = request.form.get('age')
        department = request.form.get('department')
        hospital = request.form.get('hospital')
        note = request.form.get('note', '')
        image = request.files.get('image')

        if not all([name, age, department, hospital]):
            flash('⚠️ Vui lòng điền đủ thông tin bắt buộc.')
            return redirect(url_for('index'))

        image_path = ''
        if image and image.filename != '':
            filename = secure_filename(image.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(save_path)
            image_path = f"/{save_path}"
        print([name , department] , file=sys.stderr )
        new_doctor  =  Doctor(fullname = name ,  age = age ,  deparment = department ,  hospital = hospital , note = note , image_path = image_path)
        db.session.add(new_doctor)
        db.session.commit()
        flash('✅ Đã ghi nhận thông tin bác sĩ thành công!')
    except Exception as e:
        flash(f'❌ Lỗi khi thêm bác sĩ: {e}')
    return redirect(url_for('index'))

# ====== ROUTE: Xem dữ liệu ======
@bp.route('/view', methods=['GET'])
def view_data():
    try:
        stmt = select(Doctor)
        rows = db.session.execute(stmt)
        result = rows.scalars().all()
        records = []
        for row in result :
            records.append(row.get_info())
        total_doctors = len(records)
        departments = len(set(d['department'] for d in records if d.get('department')))
        hospitals = len(set(d['hospital'] for d in records if d.get('hospital')))
    except Exception as e:
        records, total_doctors, departments, hospitals = [], 0, 0, 0
        flash(f'Lỗi khi lấy dữ liệu: {e}')
    return render_template('import_data/view.html',
                           doctors=records,
                           total_doctors=total_doctors,
                           departments=departments,
                           hospitals=hospitals)

@bp.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        flash('⚠️ Vui lòng chọn file trước khi tải lên.')
        return redirect(url_for('index'))
    
    filename = file.filename
    ext = os.path.splitext(filename)[-1].lower()
    
    try:
      if ext == '.csv':
        df = pd.read_csv(file)
      elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file)
      else:
        flash('❌ Định dạng file không được hỗ trợ.')
        return redirect(url_for('index'))
      
      df.columns = [col.strip().lower() for col in df.columns]

      success, failed = 0, 0
      for _, row in df.iterrows():
        try:
          # 🧩 Kiểm tra & làm sạch link ảnh
          image_url = str(row.get('image_url', '')).strip()
          if not image_url.startswith(('http://', 'https://')):
              image_url = "" 

          data = {
            'name' : row.get('name'),
            'age' : row.get('age'),
            'department' : row.get('department'),
            'hospital' : row.get('hospital'),
            'note' : row.get('note', ''),
            'image_url' : image_url
          }

          # ⚠️ Bỏ qua dòng thiếu thông tin bắt buộc
          if not all([data['name'], data['age'], data['department'], data['hospital']]):
              failed += 1
              continue

          #add to database
          new_doctor =  Doctor(fullname = data['name'] ,  age = data['age'] , deparment = data['department'] ,  hospital = data['hospital'] , note = data['note'] , image_path = data['image_url'])
          db.session.add(new_doctor)
          db.session.commit()
          success += 1
        except Exception as e:
            print(f"Lỗi ở dòng: {e}")  # để debug
            failed += 1
          
           # Thông báo kết quả
      msg = f"✅ Đã nhập {success} dòng thành công."
      if failed > 0:
        msg += f" ⚠️ Bỏ qua {failed} dòng lỗi."
      flash(msg)
    except Exception as e:
      flash(f'❌ Lỗi khi đọc file: {e}')
    
    return redirect(url_for('index'))

@bp.route('/delete/<int:id>', methods=['POST'])
def delete_doctor(id):
  try:
    with app.app_context():
      user_to_delete =  Doctor.query.get(id)
    if user_to_delete :
      db.session.delete(user_to_delete)
      db.session.commit()
      flash('✅ Bác sĩ đạ bị xóa!')
    else:
      flash('❌ Id bác sĩ là không tồn tại!')
  except Exception as e:
    flash(f'❌ Lỗi khi xóa bác sĩ: {e}')
  return redirect(url_for('import_data.view_data'))

@bp.route('/update', methods=['POST'])
def update_doctor():
    id = int(request.form.get('row'))
    data = {
      'name' : request.form.get('name'),
      'age' : request.form.get('age'),
      'department' : request.form.get('department'),
      'hospital' : request.form.get('hospital'),
    }
    if not all([data['name'] ,  data['age'] , data['department'] , data['hospital']]):
      flash(f'❌ All field is required!')
      return redirect(url_for('import_data.view_data'))
    try:
      doctor =  Doctor.query.get(id)
      if doctor :
        doctor.fullname = data['name']
        doctor.age = data['age']
        doctor.deparment  = data['department']
        doctor.hospital = data['hospital']
        db.session.commit()
        msg = '✅ Đã cập nhật thông tin bác sĩ thành công!'
      else:
        msg = '❌ Id bác là không được tìm thấy'
      flash(msg)
    except Exception as e:
      flash(f'❌ Lỗi khi cập nhật: {e}')
    return redirect(url_for('import_data.view_data'))


