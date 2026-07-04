# TrafficVision: AI-Powered Traffic Image Assessment Platform 🚗🚦

TrafficVision is a modern web application designed for AI-powered traffic safety analysis and urban planning. It utilizes a deep learning model to accurately classify key traffic objects from static images through an elegant, premium web interface.

The system features a three-tier architecture with a Flask web frontend for responsive, browser-based access, a FastAPI backend for high-performance model inference, and SQLite for persistent analysis history.

## ✨ Key Features and Functions

The TrafficVision platform provides comprehensive tools for traffic analysis with an intuitive web experience:

- **Core Traffic Classification**: The system classifies eight critical traffic object types: person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign.
- **Single Image Analysis**: Upload and analyze individual traffic images with instant classification results and confidence scoring.
- **Batch Processing**: Efficiently process multiple traffic images simultaneously and review results with grouped display.
- **Adjustable Confidence Threshold**: Configure detection sensitivity in real-time to fine-tune results.
- **Analysis History**: Browse, review, and manage all completed analyses with full image previews and metadata.
- **Premium Web UI**: Modern, responsive design with smooth interactions and polished visual hierarchy across all pages.

## 🧠 AI Model and Architecture

### Model Details

The system's core classification capability is built using Transfer Learning.

- **Architecture**: EfficientNet-B1
- **Training Foundation**: Pre-trained on the COCO2017 subset
- **Performance Goal**: Designed to achieve minimum 90% mean Average Precision (mAP) with processing times under 5 seconds per image.
- **Model File**: Requires `efficientnet_b1_8class_multilabel_BEST.ckpt` in the `server/` directory.

### Technology Stack

| Component | Tier | Technology | Role |
|-----------|------|------------|------|
| Client | Presentation | Flask + Jinja2 | Modern web interface with responsive, premium design |
| Server | Logic | FastAPI | Asynchronous API for model inference and business logic |
| Database | Data | SQLite | Storage for analysis history and classification results |
| ML Framework | ML | MindSpore + MindCV | Deep learning inference engine |

## 💻 Installation and Setup

### 1. Clone Repository and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/allendalangin/TrafficVision.git
cd TrafficVision

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required Python packages
pip install flask fastapi uvicorn python-multipart pydantic httpx mindspore mindcv
```

### 2. File Setup

Ensure the model checkpoint is correctly placed:
- Place `efficientnet_b1_8class_multilabel_BEST.ckpt` in the `server/` directory.

## ▶️ Running the Application

The system uses a client-server architecture and requires two separate terminal processes.

### 1. Start the FastAPI Server (Backend)

Open a terminal in the project root directory and run:

```bash
uvicorn server.api_server:app --reload --host 127.0.0.1 --port 8000
```

The backend will be available at `http://127.0.0.1:8000`.

### 2. Start the Flask Web App (Frontend)

Open a new terminal window in the project root directory and run:

```bash
python app.py
```

The web interface will be available at `http://127.0.0.1:5000`. Simply open this URL in your browser to access the application.



# Build the Flask image
docker build -f Dockerfile.flask -t my-flask-app .

# Run the Flask container
docker run -d -p 5000:5000 --name flask-container my-flask-app



# Build the FastAPI image
docker build -f Dockerfile.fastapi -t my-fastapi-app .

# Run the FastAPI container
docker run -d -p 8000:8000 --name fastapi-container my-fastapi-app
