# PCA Vision Web App

Web application for Principal Component Analysis (PCA) featuring Image Compression, Exploratory Data Analysis (EDA), and Face Recognition (Eigenface Subspace Method - ESM).

## Features

- **Image Compression:** Upload color/grayscale photos and compress them using $k$ principal components.
- **EDA (Exploratory Data Analysis):** Interactive charts showing Cumulative Explained Variance to help determine the optimal $k$ value.
- **Face Recognition (ESM):** Compare two faces (e.g., childhood vs. adult) by projecting them into an Eigenface subspace trained on a face dataset.

## Tech Stack

- **Backend:** FastAPI (Python), NumPy, OpenCV, Scikit-Learn.
- **Frontend:** React (Vite, TypeScript), Tailwind CSS, Recharts, Lucide Icons.

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js & npm

### Backend Setup
1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn numpy opencv-python scikit-learn python-multipart pillow
   ```
3. Run the server:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`.

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   The web app will be available at `http://localhost:5173`.

## Usage
1. **Compression Tab:** Upload an image, adjust the $k$ slider, and click "Run Compression". Check the EDA chart to see how much variance is captured.
2. **ESM Tab:** Click "Train Subspace" (uses Olivetti faces dataset). Upload two face photos and click "Compare Faces" to see the similarity score.

## Author
Developed for Aljabar Linear & Matriks project.
