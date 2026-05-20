import io
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

def compress_image(image_bytes, k):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape
    
    reconstructed_channels = []
    variance_ratios = []
    
    for i in range(3):
        channel = img_rgb[:, :, i]
        reconstructed, v_ratio = pca_compress_2d(channel, k)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        reconstructed_channels.append(reconstructed)
        variance_ratios.append(v_ratio)
    
    # Use average variance ratio for simplicity in EDA
    avg_variance_ratio = np.mean(variance_ratios, axis=0).tolist()
    
    reconstructed_img = np.stack(reconstructed_channels, axis=2)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR))
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    
    return encoded_img, avg_variance_ratio

# =====================================================
# Endpoints
# =====================================================

@app.post("/api/compress")
async def api_compress(file: UploadFile = File(...), k: int = Form(...)):
    try:
        contents = await file.read()
        encoded_img, variance_data = compress_image(contents, k)
        return {
            "compressed_image": f"data:image/png;base64,{encoded_img}",
            "variance_data": variance_data[:100] # Limit to top 100 for chart
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

@app.post("/api/esm/compare")
async def compare_faces(child_file: UploadFile = File(...), adult_file: UploadFile = File(...)):
    global esm_model
    if esm_model is None:
        raise HTTPException(status_code=400, detail="ESM model not trained. Call /api/esm/train first.")
    
    try:
        def process_face(file):
            contents = file.file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (64, 64))
            return img_resized.flatten() / 255.0

        child_flat = process_face(child_file)
        adult_flat = process_face(adult_file)
        
        # Project into subspace
        child_proj = esm_model.transform([child_flat])
        adult_proj = esm_model.transform([adult_flat])
        
        # Calculate distance
        dist = np.linalg.norm(child_proj - adult_proj)
        
        # Convert distance to a simple "similarity" score (0 to 100)
        # Using a simple heuristic for Olivetti space distance
        similarity = max(0, 100 - (dist * 10)) 
        
        return {
            "similarity": round(float(similarity), 2),
            "distance": round(float(dist), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
