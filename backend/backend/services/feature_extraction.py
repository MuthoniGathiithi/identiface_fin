"""
Feature extraction using InsightFace embeddings.
"""
import numpy as np
import cv2
from .face_model import get_face_model

def extract_embedding(image, bbox=None, landmarks=None):
    """
    Extract facial embedding from an image using InsightFace.
    
    Args:
        image: numpy array (BGR format)
        bbox: optional bounding box (x1, y1, x2, y2)
        landmarks: optional facial landmarks
        
    Returns:
        numpy array of 512-dimensional embedding (InsightFace default)
    """
    model = get_face_model()
    
    # InsightFace expects BGR format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Detect and extract embedding
    faces = model.get(image)
    
    if faces:
        # Return the first face's embedding
        return faces[0].embedding
    return None

def extract_embedding_from_normalized(face_image):
    """
    Extract embedding from a normalized face image.
    
    Args:
        face_image: normalized face image (numpy array, BGR format)
        
    Returns:
        numpy array of embedding
    """
    model = get_face_model()
    
    # InsightFace can work with cropped faces
    if len(face_image.shape) == 2:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
    
    faces = model.get(face_image)
    
    if faces:
        return faces[0].embedding
    return None

def extract_embeddings_from_image(image):
    """
    Extract embeddings for all faces in an image.
    
    Args:
        image: numpy array (BGR format)
        
    Returns:
        list of tuples: (bbox, embedding, landmarks)
    """
    model = get_face_model()
    
    # InsightFace expects BGR format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Detect all faces
    faces = model.get(image)
    
    results = []
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        landmarks = face.kps if hasattr(face, 'kps') else None
        results.append((bbox, embedding, landmarks))
    
    return results

def extract_embeddings_from_base64_images(base64_images):
    """
    Extract embeddings from multiple base64 encoded images.
    
    Args:
        base64_images: list of base64 encoded image strings
        
    Returns:
        list of embeddings (numpy arrays)
    """
    import base64
    from PIL import Image
    import io
    
    embeddings = []
    
    for base64_string in base64_images:
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array (BGR for OpenCV)
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            # Convert RGB to BGR
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        embedding = extract_embedding(image_array)
        if embedding is not None:
            embeddings.append(embedding)
    
    return embeddings
