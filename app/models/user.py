import string
from sqlmodel import Field, SQLModel ,create_engine
from datetime import date, datetime

class User(SQLModel , table = True):
    id: int | None = Field(default=None, primary_key=True)
    email: string 
    hashed_password: string
    full_name = string
    is_active : bool = False
    created_at : datetime = Field(default_factory=datetime.utcnow, nullable=False)

    
    
    
    