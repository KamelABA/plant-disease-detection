# 🌿 PlantDetect - AI-Powered Plant Disease Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.0+-06B6D4.svg)

## 📖 Overview

PlantDetect is an AI-powered web application that helps farmers and home gardeners identify plant diseases quickly and accurately. Using advanced machine learning models, the system can detect diseases in various crops including:

- 🍅 **Tomato** - 10 disease types
- 🥔 **Potato** - 3 disease types
- 🍇 **Grape** - 4 disease types
- 🌽 **Corn/Maize** - 4 disease types
- 🧶 **Cotton** - 4 disease types
- 🍎 **Apple** - 4 disease types

## ✨ Features

- 🔍 **Disease Detection**: Upload a leaf image and get instant disease diagnosis
- 📊 **Comprehensive Database**: Access detailed information about plant diseases
- 💊 **Treatment Recommendations**: Get supplement and treatment suggestions
- 📜 **History Tracking**: Keep track of your previous disease detections
- 🌐 **Multi-language Support**: Available in English and Arabic
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **MongoDB Atlas** - Cloud database
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing

### Frontend
- **HTML5**
- **Tailwind CSS** - Utility-first CSS framework
- **Bootstrap 5** - UI components
- **JavaScript** - Interactive features

### Machine Learning Models
- Pre-trained CNN models (.h5 files) for each crop type
- Image preprocessing with OpenCV
- Real-time prediction with TensorFlow

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- MongoDB Atlas account (or local MongoDB installation)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/KamelABA/nebtetek.git
   cd nebtetek
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r req.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   MONGODB_URI=your_mongodb_connection_string
   SECRET_KEY=your_secret_key
   ```

5. **Run the application**
   ```bash
   python app2.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
nebtetek/
├── app2.py                 # Main Flask application
├── HtmlPage/               # HTML templates
│   ├── home.html          # Homepage
│   ├── login.html         # Login page
│   ├── signup.html        # Registration page
│   ├── leaf-detection.html # Disease detection page
│   ├── Output.html        # Results display
│   └── ...                # Other templates
├── static/                 # Static assets
│   ├── assets/            # Images and media
│   ├── img/               # Icons and logos
│   ├── admin/             # Admin dashboard assets
│   └── *.css              # Stylesheets
├── *.h5                   # Pre-trained ML models
├── req.txt                # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### MongoDB Atlas Setup

1. Create a free MongoDB Atlas account at [mongodb.com](https://www.mongodb.com/cloud/atlas)
2. Create a new cluster
3. Create a database user with read/write permissions
4. Whitelist your IP address
5. Get your connection string and update the `.env` file

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `SECRET_KEY` | Flask secret key for sessions |

## 🚀 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Homepage |
| `/login` | GET | Login page |
| `/signup` | GET | Registration page |
| `/adduser` | POST | Create new user |
| `/checklogin` | POST | Authenticate user |
| `/leaf-detection` | GET | Disease detection page |
| `/predictionpotato` | POST | Potato disease prediction |
| `/predictiontomato` | POST | Tomato disease prediction |
| `/predictiongrape` | POST | Grape disease prediction |
| `/predictioncotton` | POST | Cotton disease prediction |
| `/predictioncorn` | POST | Corn disease prediction |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Kamel** - *Initial work* - [KamelABA](https://github.com/KamelABA)

## 🙏 Acknowledgments

- TensorFlow team for the amazing deep learning framework
- MongoDB Atlas for cloud database services
- Tailwind CSS for the utility-first CSS framework
- All contributors and supporters of this project

---

<p align="center">
  Made with ❤️ for a greener future
</p>
