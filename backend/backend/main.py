from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import Base, engine
from .routes import auth, classes, students, attendance

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Facial Recognition Attendance System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(classes.router)
app.include_router(students.router)
app.include_router(attendance.router)

@app.get("/")
def root():
    return {"message": "Facial Recognition Attendance System API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

