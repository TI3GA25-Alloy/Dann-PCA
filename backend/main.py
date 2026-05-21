import io
import time
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA as SklearnPCA
    
app = FastAPI()

# Allow CORS for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for ESM
esm_model = None
esm_mean_face = None

# =====================================================
# PCA Logic (Refactored from PCA.py)
# =====================================================

def pca_compress_2d(X, k):
    X = X.astype(np.float64)
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    k = min(k, len(S))
    X_reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :] + mean_vec
    
    # Variance ratio for EDA
    total_variance = np.sum(S**2)
    explained_variance = np.cumsum(S**2) / total_variance
    
    return X_reconstructed, explained_variance.tolist()

def compress_image(image_bytes, k, grayscale=False):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    
    # Convert to grayscale if requested
    if grayscale:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        reconstructed, v_ratio = pca_compress_2d(img_gray, k)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        avg_variance_ratio = v_ratio
        reconstructed_img = cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        reconstructed_channels = []
        variance_ratios = []
        
        for i in range(3):
            channel = img_rgb[:, :, i]
            reconstructed, v_ratio = pca_compress_2d(channel, k)
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
            reconstructed_channels.append(reconstructed)
            variance_ratios.append(v_ratio)
        
        avg_variance_ratio = np.mean(variance_ratios, axis=0).tolist()
        reconstructed_img = cv2.cvtColor(np.stack(reconstructed_channels, axis=2), cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode('.png', reconstructed_img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    
    return encoded_img, avg_variance_ratio

# =====================================================
# Endpoints
# =====================================================

@app.post("/api/compress")
async def api_compress(
    file: UploadFile = File(...), 
    k: int = Form(...), 
    grayscale: bool = Form(False)
):
    start_time = time.time()
    try:
        contents = await file.read()
        encoded_img, variance_data = compress_image(contents, k, grayscale)
        processing_time = (time.time() - start_time) * 1000 # ms
        return {
            "compressed_image": f"data:image/png;base64,{encoded_img}",
            "variance_data": variance_data[:100], # Limit to top 100 for chart
            "processing_time": round(processing_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/esm/train")
async def train_esm():
    global esm_model, esm_mean_face
    try:
        data = fetch_olivetti_faces()
        faces = data.data  # shape (400, 4096) - 64x64 faces
        esm_model = SklearnPCA(n_components=100, whiten=True)
        esm_model.fit(faces)
        esm_mean_face = esm_model.mean_
        return {"status": "success", "message": "ESM model trained on Olivetti faces dataset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/esm/eigenfaces")
async def get_eigenfaces(n: int = 10):
    global esm_model
    if esm_model is None:
        raise HTTPException(status_code=400, detail="ESM model not trained.")
    
    try:
        eigenfaces = []
        for i in range(min(n, esm_model.n_components_)):
            # Normalize component to 0-255 for visualization
            comp = esm_model.components_[i].reshape(64, 64)
            comp_norm = cv2.normalize(comp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, buffer = cv2.imencode('.png', comp_norm)
            encoded = base64.b64encode(buffer).decode('utf-8')
            eigenfaces.append(f"data:image/png;base64,{encoded}")
        
        return {"eigenfaces": eigenfaces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_face_aligned(contents):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    detected = False
    if len(faces) > 0:
        # Take the largest face
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_img = gray[y:y+h, x:x+w]
        detected = True
    else:
        # Fallback to center crop if no face detected
        h, w = gray.shape
        side = min(h, w)
        face_img = gray[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]

    # Resize to match Olivetti dataset (64x64)
    img_resized = cv2.resize(face_img, (64, 64))
    return img_resized.flatten() / 255.0, detected

@app.post("/api/esm/compare")
async def compare_faces(child_file: UploadFile = File(...), adult_file: UploadFile = File(...)):
    global esm_model
    if esm_model is None:
        raise HTTPException(status_code=400, detail="ESM model not trained. Call /api/esm/train first.")
    
    start_time = time.time()
    try:
        child_contents = await child_file.read()
        adult_contents = await adult_file.read()

        child_flat, child_detected = process_face_aligned(child_contents)
        adult_flat, adult_detected = process_face_aligned(adult_contents)
        
        # Project into subspace
        child_proj = esm_model.transform([child_flat])
        adult_proj = esm_model.transform([adult_proj]) if 'adult_proj' in locals() else esm_model.transform([adult_flat])
        
        # Calculate distance
        dist = np.linalg.norm(child_proj - adult_proj)
        
        # Heuristic similarity: Olivetti space distance usually 5-25
        similarity = max(0, 100 - (dist * 8)) 
        
        processing_time = (time.time() - start_time) * 1000 # ms
        
        return {
            "similarity": round(float(similarity), 2),
            "distance": round(float(dist), 4),
            "processing_time": round(processing_time, 2),
            "child_detected": child_detected,
            "adult_detected": adult_detected
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
