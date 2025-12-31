"""
InsightFace model initialization and management.
This module provides a singleton pattern for the InsightFace model.
"""
import insightface
import numpy as np
import os

class FaceModel:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceModel, cls).__new__(cls)
        return cls._instance
    
    def get_model(self):
        """Get or initialize the InsightFace model."""
        if self._model is None:
            # Initialize InsightFace model
            # Using BUFFALO_L model for better accuracy
            self._model = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']  # Use CPU, can be changed to CUDA if available
            )
            self._model.prepare(ctx_id=0, det_size=(640, 640))
        return self._model

# Global instance
face_model = FaceModel()

def get_face_model():
    """Get the global InsightFace model instance."""
    return face_model.get_model()

