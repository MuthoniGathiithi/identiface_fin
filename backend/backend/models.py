from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import json

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    classes = relationship("Class", back_populates="lecturer")

class Class(Base):
    __tablename__ = "classes"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    time = Column(String, nullable=False)
    lecturer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    lecturer = relationship("User", back_populates="classes")
    students = relationship("Student", back_populates="class_obj")
    attendance_sessions = relationship("AttendanceSession", back_populates="class_obj")

class Student(Base):
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    class_obj = relationship("Class", back_populates="students")
    embeddings = relationship("StudentEmbedding", back_populates="student")
    attendance_records = relationship("AttendanceRecord", back_populates="student")

class StudentEmbedding(Base):
    __tablename__ = "student_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    pose = Column(String, nullable=False)  # 'right', 'left', 'up', 'down'
    embedding = Column(Text, nullable=False)  # JSON array of floats
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    student = relationship("Student", back_populates="embeddings")
    
    def get_embedding_array(self):
        return json.loads(self.embedding)
    
    def set_embedding_array(self, arr):
        self.embedding = json.dumps(arr)

class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    session_date = Column(DateTime(timezone=True), server_default=func.now())
    
    class_obj = relationship("Class", back_populates="attendance_sessions")
    records = relationship("AttendanceRecord", back_populates="session")

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    status = Column(String, nullable=False)  # 'PRESENT' or 'ABSENT'
    
    session = relationship("AttendanceSession", back_populates="records")
    student = relationship("Student", back_populates="attendance_records")

