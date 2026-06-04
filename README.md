# Defect-AI: Product Defect Detection System

## Overview
Defect-AI is a browser-based application designed to perform real-time product defect detection using computer vision techniques. The system utilizes a smartphone or webcam to capture live video and analyze it to identify defects in products. It provides immediate and interpretable feedback, making it suitable for quality inspection in manufacturing environments.

---

## Features
- Real-time inspection using a camera  
- Automated defect detection using AI techniques  
- Instant classification of products as pass or fail  
- Modular design for easy customization and extension  
- Runs entirely in the browser without requiring complex setup  
- Compatible with both desktop and mobile devices  

---

## Technology Stack
- Frontend: HTML, CSS, JavaScript  
- Computer Vision: TensorFlow.js or custom image processing logic  
- Camera Integration: Web APIs (MediaDevices)  
- Deployment: GitHub Pages or local hosting  

---

## Project Structure
```
Defect-AI/
│── index.html          # Main user interface
│── app.js              # Application controller
│── camera.js           # Camera handling module
│── detector.js         # Defect detection logic
│── inspector.js        # Inspection workflow
│── preprocessing.js    # Image preprocessing utilities
│── training.js         # Model training logic
│── storage.js          # Data handling and storage
│── styles.css          # Styling and layout
```

---

## Installation and Setup

### Prerequisites
- A modern web browser (Chrome, Edge, Firefox)  
- Visual Studio Code (recommended)  

### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/username/defect-ai.git
   ```

2. Open the project folder in Visual Studio Code  

3. Install the **Live Server** extension  

4. Right-click on `index.html` and select **Open with Live Server**

Alternatively, open `index.html` directly in a web browser.

---

## Usage
1. Launch the application in a browser  
2. Grant camera access when prompted  
3. Position the product within the camera frame  
4. The system will display:
   - Pass: No defect detected  
   - Fail: Defect detected with classification  

---

## System Workflow
- Capture video frames from the camera  
- Perform preprocessing such as resizing and normalization  
- Apply detection logic or trained model  
- Generate classification results  
- Display output in real time  

---

## Future Enhancements
- Integration of advanced deep learning models  
- Expansion of supported defect categories  
- Cloud-based model training and storage  
- Mobile application deployment  
- Integration with IoT systems for automated alerts  

---

## Limitations
- Performance depends on lighting and camera quality  
- Accuracy is limited by the training data and model capability  
- Requires stable positioning of the product during inspection  

---

## Author
Akash P
