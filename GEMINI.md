# PCA Vision Web App

A full-stack web application for Principal Component Analysis (PCA) featuring Image Compression, Exploratory Data Analysis (EDA), and Face Recognition using the Eigenface Subspace Method (ESM).

## Project Overview

- **Purpose:** Educational tool for visualizing and applying PCA concepts to image processing and face recognition.
- **Backend:** FastAPI (Python) handles the PCA computation, image processing (OpenCV/NumPy), and ESM training (scikit-learn).
- **Frontend:** React (TypeScript/Vite) provides an interactive UI with real-time compression sliders and variance charts (Recharts).
- **Core Features:**
  - **Image Compression:** Compress images by selecting the number of principal components ($k$).
  - **EDA:** Visualize Cumulative Explained Variance to understand information retention.
  - **ESM:** Project face images into an "Eigenface" subspace (trained on the Olivetti dataset) to compare similarity.

## Directory Structure

- `/backend`: Python FastAPI application.
- `/frontend`: React/TypeScript application.
- `/conductor`: Implementation plans and tracking.

## Development Commands

### Backend

- **Directory:** `backend/`
- **Run Server:** `python main.py` (Starts FastAPI on `http://localhost:8000`)
- **Dependencies:** `fastapi`, `uvicorn`, `numpy`, `opencv-python`, `scikit-learn`, `python-multipart`, `pillow`.

### Frontend

- **Directory:** `frontend/`
- **Install:** `npm install`
- **Run Dev:** `npm run dev` (Starts Vite on `http://localhost:5173`)
- **Build:** `npm run build`
- **Lint:** `npm run lint`

## Architecture & Conventions

### Backend Logic
- PCA logic for 2D images is implemented using SVD (`np.linalg.svd`).
- For RGB images, PCA is performed independently on each channel.
- ESM uses `sklearn.decomposition.PCA` with whitening enabled.

### API Endpoints
- `POST /api/compress`: Takes an image and integer `k`. Returns Base64 compressed image and variance data.
- `POST /api/esm/train`: Initializes the ESM model using the Olivetti faces dataset.
- `POST /api/esm/compare`: Takes two images (`child_file`, `adult_file`), projects them into the subspace, and returns a similarity score.

### Frontend Conventions
- **Styling:** Tailwind CSS.
- **Charts:** Recharts for EDA visualization.
- **Icons:** Lucide React.
- **State Management:** Standard React `useState`/`useEffect` hooks.

## Verification Workflow

1. **Backend:** Ensure Python 3.10+ is used. Test endpoints using Swagger UI at `/docs`.
2. **Frontend:** Verify Tailwind integration and Vite HMR. Ensure Recharts correctly renders the `variance_data` array returned from the backend.
3. **Integration:** CORS is allowed for all origins in `main.py` to facilitate local development.
