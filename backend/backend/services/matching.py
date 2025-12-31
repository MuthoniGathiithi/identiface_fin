import numpy as np
from typing import List, Tuple

def calculate_distance(embedding1, embedding2):
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        embedding1: numpy array
        embedding2: numpy array
        
    Returns:
        float: distance
    """
    return np.linalg.norm(embedding1 - embedding2)

def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    InsightFace embeddings are typically normalized, so cosine similarity works well.
    
    Args:
        embedding1: numpy array
        embedding2: numpy array
        
    Returns:
        float: similarity score (0-1, higher is more similar)
    """
    # Normalize embeddings
    emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
    emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
    
    # Cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    
    # Normalize to 0-1 range (cosine similarity is already -1 to 1)
    return (similarity + 1) / 2

def match_embedding(query_embedding, stored_embeddings, threshold=0.5):
    """
    Match a query embedding against stored embeddings.
    InsightFace embeddings are 512-dimensional.
    
    Args:
        query_embedding: numpy array (512-dim)
        stored_embeddings: list of tuples (student_id, pose, embedding_array)
        threshold: similarity threshold (default 0.5 for InsightFace)
        
    Returns:
        tuple: (matched_student_id, best_similarity) or (None, 0.0)
    """
    best_match = None
    best_similarity = 0.0
    
    for student_id, pose, embedding_array in stored_embeddings:
        embedding = np.array(embedding_array)
        similarity = calculate_similarity(query_embedding, embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = student_id
    
    if best_similarity >= threshold:
        return (best_match, best_similarity)
    
    return (None, best_similarity)

def match_multiple_embeddings(query_embeddings, stored_embeddings, threshold=0.5):
    """
    Match multiple query embeddings against stored embeddings.
    Uses voting: if multiple poses match the same student, count as one match.
    
    Args:
        query_embeddings: list of numpy arrays
        stored_embeddings: list of tuples (student_id, pose, embedding_array)
        threshold: similarity threshold
        
    Returns:
        dict: {student_id: match_count} for students that matched
    """
    matches = {}
    
    for query_emb in query_embeddings:
        student_id, similarity = match_embedding(query_emb, stored_embeddings, threshold)
        if student_id:
            if student_id not in matches:
                matches[student_id] = 0
            matches[student_id] += 1
    
    return matches

def find_best_matches(query_embeddings, stored_embeddings, threshold=0.5, min_matches=1):
    """
    Find best matching students from query embeddings.
    
    Args:
        query_embeddings: list of numpy arrays
        stored_embeddings: list of tuples (student_id, pose, embedding_array)
        threshold: similarity threshold
        min_matches: minimum number of embeddings that must match
        
    Returns:
        set: set of student_ids that matched
    """
    match_counts = match_multiple_embeddings(query_embeddings, stored_embeddings, threshold)
    
    # Return students that matched at least min_matches times
    return {student_id for student_id, count in match_counts.items() if count >= min_matches}
