from sqlalchemy import Integer, String ,Column, Boolean, DateTime , ForeignKey , Text , select 
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from . import db

class Doctor(db.Model):
    __tablename__ = "doctors"
    
    id = Column(Integer, primary_key=True)
    fullname = Column(String(100))
    age =  Column(Integer)
    deparment = Column(String(100))
    hospital = Column(String(100))
    note = Column(Text)
    image_path= Column(String(255))

    @classmethod
    def reproduce(cls):
        return cls()
    def createDoctor(self):
        db.session.add(self.reproduce())
        db.session.commit()

    def get_info(self):
        return {
            'id' :  self.id,
            'name' : self.fullname ,
            'age' : self.age , 
            'department' :self.deparment , 
            'hospital' :self.hospital , 
            'note' : self.note , 
            'image_url' : self.image_path.replace('/flaskr' , '').replace('\\' , '/') if  self.image_path else ''
        }
    def __repr__(self):
        return f'<User {self.fullname}>'
        
        