# ArcFace Face Recognition Access Control System

<div align="center">
  <img src="https://raw.githubusercontent.com/wiki/opencv/opencv/logo/OpenCV_logo_no_text.png" height="80">
  <img src="https://insightface.ai/assets/img/custom/logo3.jpg" height="80">
  <br><br>
  
  [![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
  [![InsightFace](https://img.shields.io/badge/Model-ArcFace-orange)](https://github.com/deepinsight/insightface)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

<p align="center">A high-accuracy face recognition system using ArcFace for access control applications with an emphasis on security, reliability, and ease of use.</p>

---

## 🌟 Key Features

- **State-of-the-art facial recognition** using ArcFace model (99.82% accuracy on LFW benchmark)
- **Real-time processing** optimized for access control applications  
- **Modular architecture** with separate components for detection, embedding, and verification
- **Comprehensive database management** tools for adding, removing, and updating identities
- **Multiple database options** including standard storage and FAISS for high-scale deployments
- **Detailed logging and access records** for security auditing
- **Extensive configuration options** to balance security and convenience
- **Simple command-line interface** for all operations

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration Options](#-configuration-options)
- [Database Management](#-database-management)
- [Technical Architecture](#-technical-architecture)
- [Recognition Performance](#-recognition-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 💻 System Requirements

### Minimum Requirements
- Python 3.7+
- 4GB RAM
- Intel/AMD x86-64 CPU (2 cores+)
- 500MB disk space
- Webcam (for live recognition)

### Recommended Requirements
- Python 3.8+
- 8GB RAM
- Modern Intel/AMD CPU (4+ cores)
- NVIDIA GPU with CUDA support
- 1GB+ disk space
- HD webcam or IP camera

### Tested Platforms
- Windows 10/11
- Ubuntu 20.04/22.04
- macOS 11+

## 🚀 Installation

### From GitHub

```bash
# Clone the repository
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### With GPU Support (Optional)

```bash
# Install GPU versions of libraries for better performance
pip install onnxruntime-gpu faiss-gpu
```

## 🏁 Quick Start

### 1. Create Face Database

First, organize your face images into folders named after each person:

```
face_data/
├── John_Smith/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Jane_Doe/
│   ├── image1.jpg
│   └── ...
└── ...
```

Then, create the database:

```bash
python main.py create-db --root face_data
```

### 2. Test Recognition

Test with a single image:

```bash
python main.py test --image path/to/test.jpg
```

### 3. Run Live Recognition

```bash
python main.py live
```

Press 'q' to quit, 's' to take a screenshot.

## ⚙️ Configuration Options

Configuration settings are located in `config/settings.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DETECTION_THRESHOLD` | `0.6` | Face detection confidence threshold |
| `RECOGNITION_THRESHOLD` | `0.5` | Similarity threshold for recognition |
| `USE_FAISS` | `False` | Whether to use FAISS for database |
| `MODEL_NAME` | `buffalo_l` | InsightFace model name |
| `DET_SIZE` | `(640, 640)` | Detection size |
| `CTX_ID` | `0` | CUDA device ID (`0` for first GPU, `-1` for CPU) |

### Using Command-Line Options

You can override configuration options directly from the command line:

```bash
# Adjust recognition threshold for stricter matching
python main.py live --recognition-threshold 0.65

# Use FAISS for larger databases
python main.py live --use-faiss

# Use CPU for detection (when no GPU available)
python main.py live --ctx-id -1
```

## 📊 Database Management

The system includes comprehensive tools for managing face databases:

```bash
# List all identities
python tools/manage_database.py list

# Add a new person from multiple images
python tools/manage_database.py add --name "John Smith" --folder path/to/john_photos

# Remove a person
python tools/manage_database.py remove --name "John Smith"

# Update an existing person's face data
python tools/manage_database.py update --name "Jane Doe" --folder path/to/new_photos

# Backup the database
python tools/manage_database.py backup
```

### Best Practices for Database Management

- Use 5-10 high-quality images per person from different angles
- Update embeddings when a person's appearance changes significantly
- Create regular backups, especially before major updates
- Test recognition accuracy periodically with known samples
- For large databases (500+ identities), switch to FAISS

## 🏗️ Technical Architecture

The system follows a modular design with clear separation of concerns:

```
face_recognition_system/
├── config/               # Configuration settings
├── src/                  # Source code modules
│   ├── database/         # Database operations
│   ├── face/             # Face detection and embedding
│   ├── utils/            # Utility functions
│   └── access_control/   # Access verification
├── tools/                # Command-line tools
└── main.py               # Main entry point
```

### Recognition Pipeline

1. **Face Detection**: Using RetinaFace from InsightFace
2. **Face Alignment**: Based on detected facial landmarks
3. **Embedding Generation**: Using ArcFace (ResNet-50) generating 512D embeddings
4. **Embedding Comparison**: Using cosine similarity for matching
5. **Access Decision**: Based on similarity threshold

### Database Storage Options

1. **Standard Database**: Pickle-based storage for small to medium deployments
   - Fast for databases with up to ~500 identities
   - Simple backup and restore

2. **FAISS Database**: For large-scale deployments
   - Optimized for databases with 500+ identities
   - Significant performance advantage for large databases
   - More complex backup and restore processes

## 🎯 Recognition Performance

### Model Details

The system uses ArcFace with the following specifications:

- **Model**: InsightFace's `buffalo_l` (ResNet-50 backbone)
- **Embedding Size**: 512-dimensional vector
- **Comparison Method**: Cosine similarity with normalization
- **LFW Benchmark**: 99.82% accuracy

### Tuning Recognition Parameters

| Threshold | False Accepts | False Rejects | Typical Use Case |
|-----------|---------------|---------------|------------------|
| 0.3-0.4   | Higher        | Lower         | Convenience-focused |
| 0.5       | Balanced      | Balanced      | General usage |
| 0.6-0.7   | Lower         | Higher        | Security-focused |

For high-security applications, increase the recognition threshold to 0.65 or higher.

### Performance Benchmarks

| Database Size | Standard DB | FAISS DB |
|---------------|-------------|----------|
| 10 identities | 2ms         | 5ms      |
| 100 identities | 15ms       | 7ms      |
| 1000 identities | 130ms     | 10ms     |

*Measured on Intel i7 CPU with 16GB RAM. GPU acceleration improves times by 40-60%.*

## ❓ Troubleshooting

### Common Issues

#### OpenCV GUI Error

```
Error: The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
```

**Solution**: Install OpenCV with GUI support:
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

#### Face Not Detected

If faces aren't being detected properly:

1. Check lighting conditions - ensure even, bright lighting
2. Adjust `DETECTION_THRESHOLD` in settings (lower for better detection)
3. Verify camera is working correctly
4. Test with different `DET_SIZE` values

#### Poor Recognition Accuracy

If recognition results are unreliable:

1. Add more varied training images for the person
2. Ensure face is clearly visible in the training images
3. Check for database quality issues (use `tools/manage_database.py test`)
4. Adjust `RECOGNITION_THRESHOLD` based on your needs

### Getting Help

If you encounter issues:
1. Check the logs in the `logs/` directory
2. Open an issue on GitHub with detailed information
3. Include log files and system information

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the ArcFace implementation
- [OpenCV](https://opencv.org/) for computer vision utilities
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- All contributors who have helped improve this system

---

<p align="center">
  Developed with ❤️ for advanced, ethical face recognition
</p>
