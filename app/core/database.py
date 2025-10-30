from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL
from config import Settings , userdb , passworddb

settings = Settings()
url_db = URL.create(
    "postgresql",
    username=userdb,
    password=passworddb,  # plain (unescaped) text
    host="localhost",
    database="agent_ai",
)
engine = create_engine(url_db , pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()