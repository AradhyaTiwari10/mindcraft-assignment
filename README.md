# PII Masking System

A full-stack application that detects and masks Personally Identifiable Information (PII) from uploaded images like Aadhaar cards.

## 🎯 Features

- **Image Upload**: Support for JPG/PNG image uploads
- **OCR Processing**: Text extraction using EasyOCR
- **PII Detection**: Identifies names, addresses, phone numbers, Aadhaar numbers, DOB, and emails
- **Image Masking**: Redacts PII with black boxes or blur effects
- **Real-time Preview**: See original and masked images side by side

## 🛠️ Tech Stack

### Backend
- **Python 3** with FastAPI
- **EasyOCR** for text extraction
- **OpenCV** for image processing
- **Regex patterns** for PII detection
- **Pillow** for image manipulation

### Frontend
- **React.js** with modern hooks
- **Axios** for API communication
- **TailwindCSS** for styling
- **File upload** with drag & drop support

## 📁 Project Structure

```
mindcraft-assignment/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── ocr.py               # OCR processing logic
│   ├── pii_detector.py      # PII detection patterns
│   ├── masker.py            # Image masking utilities
│   ├── requirements.txt     # Python dependencies
│   └── sample_images/       # Test images
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 📋 API Endpoints

- `POST /mask` - Upload image and get masked version
- `GET /health` - Health check endpoint

## 🎥 Demo Features

- Upload Aadhaar card images
- Real-time PII detection
- Side-by-side comparison
- Download masked images
- Responsive design

## 🔒 Security

- CORS enabled for frontend communication
- Input validation for image files
- Secure file handling
- No data persistence (images processed in memory)

## 📝 License

MIT License - Feel free to use for your internship application!