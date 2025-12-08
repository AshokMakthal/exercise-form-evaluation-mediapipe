# 💪 Exercise Form Analyzer (Bicep Curl + Lateral Raise)
A real-time exercise-form analysis system built using **Streamlit**, **MediaPipe**, and **OpenCV**.  
The project detects human pose, evaluates exercise form using rule-based logic, counts reps, and provides visual feedback for:

- ✔ Right-arm Bicep Curl  
- ✔ Lateral Raise (Both Arms)  

This repository includes:
- A full Streamlit UI (`humantrack.py`)
- Automated environment + dependency installation
- A launcher script (`run_project.py`) that runs the project with **one command**

---

## 🚀 Features

### 🔹 Pose Estimation (MediaPipe)
Extracts:
- Shoulders  
- Elbows  
- Wrists  
- Hips  

### 🔹 Form Correction Rules
Implemented rules include:
- Elbow angle range for bicep curls  
- Wrist–shoulder alignment for lateral raise  
- Back posture symmetry  
- Shoulder abduction angle  
- Rep counting logic for both arms  

### 🔹 Modes Available
- **Upload Video Mode** → Processes uploaded videos  
- **Webcam Mode** → Live real-time exercise analysis  

### 🔹 Output Includes:
- Real-time overlay feedback  
- Rep count (Good/Wrong)  
- Angle charts  
- Saved processed video (`demo_output.mp4`)

---

## 📁 Project Structure

```bash
/project-folder
│
├── humantrack.py          # Main Streamlit application (Contains all exercise detection logic)
├── run_project.py         # Auto-installs environment & launches app with one command
├── requirements.txt       # All required dependencies for the project
├── demo_output.mp4        # Auto-generated processed video output
└── README.md              # Project documentation
run_project.py

##Runs the whole project with one single command:

python run_project.py


Automatically:

Creates a virtual environment (venv/)

Installs all dependencies from requirements.txt

Launches the Streamlit app

Opens the browser at http://localhost:8501
