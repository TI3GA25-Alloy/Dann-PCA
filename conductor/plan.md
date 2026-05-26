# Implementation Plan: PCA Web App with EDA & ESM

## Objective
Build a web application for Principal Component Analysis (PCA) featuring:
1. **Image Compression:** Upload color/grayscale photos, compress using $k$ principal components.
2. **Exploratory Data Analysis (EDA):** Visualize eigenvalues (e.g., Scree plot / Explained Variance Ratio) to help users choose optimal $k$.
3. **Face Recognition (Eigenface Subspace Method - ESM):** Train on a dataset to build the Eigenface subspace, then project and compare two uploaded photos (e.g., child vs. adult) to compute similarity.

## Tech Stack
- **Backend:** FastAPI (Python), NumPy, OpenCV, scikit-learn.
- **Frontend:** React (Vite), Tailwind CSS, Recharts (for EDA charts).

## Implementation Steps

### Phase 1: Backend API (FastAPI)
1. **Setup FastAPI:** Initialize project structure.
2. **Compression Endpoint (`/api/compress`):**
   - Accept image upload and $k$.
   - Perform PCA per RGB channel (refactoring logic from `PCA.py`).
   - Return compressed image (Base64) and eigenvalue statistics for EDA.
3. **ESM Training Endpoint (`/api/esm/train`):**
   - Fetch/load a standard face dataset (e.g., Olivetti faces from scikit-learn) or accept ZIP upload.
   - Compute mean face, covariance, eigenvalues/vectors (Eigenfaces).
   - Store subspace model in memory.
4. **ESM Predict Endpoint (`/api/esm/compare`):**
   - Accept 2 face images (e.g., Child and Adult).
   - Project both into the Eigenface subspace.
   - Calculate Euclidean distance or Cosine similarity in the subspace.
   - Return similarity score and projection details.

### Phase 2: Frontend (React)
1. **Setup Vite + React + Tailwind.**
2. **Compression & EDA Tab:**
   - Image uploader.
   - Slider or input for $k$.
   - Before/After image viewer.
   - Scree Plot (Cumulative Variance) chart using Recharts.
3. **Face Recognition (ESM) Tab:**
   - UI to initialize/train the subspace.
   - Upload areas for "Child Photo" and "Adult Photo".
   - Result section displaying similarity score (match probability).

### Phase 3: Integration & Polish
1. Connect React to FastAPI.
2. Add loading states and error handling.
3. Finalize styling.

## Verification
- Compression works for RGB images with varying $k$.
- EDA chart accurately reflects variance data from the backend.
- ESM comparison yields higher similarity for same-person photos vs different-person photos.
