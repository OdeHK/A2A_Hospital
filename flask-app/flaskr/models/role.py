from sqlalchemy import Integer, String ,Column, Boolean, DateTime , Text , ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from . import db

class Role(db.Model):
    __tablename__ = "roles"

    id =Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    def __repr__(self):
        return f'<Role {self.name}>'