# 🏋️ Exercise Form Evaluation using MediaPipe Pose  
This project evaluates **Bicep Curl** and **Lateral Raise** exercise form using **MediaPipe Human Pose Estimation**, rule-based analysis, and time-series smoothing.

It fulfills the internship task requirements:

✔ Human pose estimation  
✔ Keypoint extraction & smoothing  
✔ Rule-based correctness logic  
✔ Frame-wise real-time feedback  
✔ Rep counting (Good vs Wrong)  
✔ Back posture analysis  
✔ Output video with overlay  
✔ Documentation + challenges + improvement ideas  

---

# 📂 Repository Structure

```
exercise-form-evaluation-mediapipe/
│
├── humantrack.py                 
├── requirements.txt              
├── README.md                     
│
├── sample_videos/
│     ├── input_video.mp4         
│     ├── demo_bicep_output.mp4    
│     ├── demo_lateral_output.mp4   
│
├── screenshots/
│     ├── curl_example.png        
│     ├── lateral_example.png      
│
└── report/
      ├── project_report.pdf       
```

---

# 🚀 Features

### 🟦 1. Bicep Curl Evaluation  
- Right-arm elbow angle tracking  
- Automatic smoothing using deque buffer  
- Rep counting based on amplitude + direction change  
- Good/Wrong rep classification  
- Back posture validation  
- Real-time on-screen feedback  

### 🟧 2. Lateral Raise Evaluation  
- Both arms evaluated independently  
- Wrist-to-shoulder height rule  
- Shoulder angle calculation  
- Back posture detection  
- Good/Wrong rep classification  
- Per-arm rep tracking  

### 🟩 3. General System Features  
- Works with **uploaded video** or **webcam**  
- Saves **processed output video**  
- Real-time skeleton + feedback overlay  
- Time-series angle visualization (Streamlit)  

---

# 📐 Pose Rules Used

### **Bicep Curl Rules**
| Rule | Description |
|------|-------------|
| Elbow angle >150° | Arm is straight (start position) |
| 100–150° | Half curl |
| 40–100° | Good curl range |
| <40° | Full contraction |
| ∆Angle + direction | Rep detection |

### **Lateral Raise Rules**
| Rule | Logic |
|------|-------|
| Wrist height near shoulder | Good raise |
| Wrist 40–80px below shoulder | Arm too low (wrong) |
| Elbow angle >150° | Arm straight (required) |
| Shoulder angle 70–100° | Valid T-pose top position |

### **Back Posture Rule**
A loose rule that detects excessive leaning:  
- Tilt <60px → Good  
- 60–120px → Acceptable  
- >120px → Wrong posture  

---

# 🧠 How Rep Counting Works

### Bicep Curl (Right Arm)
1. Detect direction (up/down) using derivative sign  
2. “Up → Down” + angle compression = rep  
3. Validate form at bottom of curl  
4. Count as Good or Wrong

### Lateral Raise (Both Arms)
1. DOWN threshold (<35°) → arm ready  
2. UP threshold (>75°) → evaluate quality  
3. DOWN again → complete rep  
4. Good/Wrong based on:
   - Wrist height  
   - Back posture  
   - Elbow straight  
   - Shoulder angle  

---

# ▶ How to Run

### Install packages:
```
pip install -r requirements.txt
```

### Run Streamlit App:
```
streamlit run humantrack.py
```

---

# 📸 Screenshots  
Add your actual images to:

```
screenshots/curl_example.png
screenshots/lateral_example.png
```

Example placeholders:

```
curl_example.png — shows bicep curl overlay
lateral_example.png — shows lateral raise overlay
```

---

# 📘 Report  
The report is available in:

📁 `report/project_report.pdf`

It includes:  
✔ Posture rules  
✔ Rep logic explanation  
✔ Challenges faced  
✔ How to handle multiple people  
✔ Improvements  

---

# 👥 Challenges Faced

### 1. Multiple People in Frame  
MediaPipe Pose returns **only one person** by default.  
Possible solutions:  
- Use **MediaPipe Holistic + multi-pose mode**  
- Use **OpenPose BODY_25** model  
- Use **YOLO person tracking + MediaPipe pose per-person**  

### 2. Occlusions / Camera Angle Variations  
Angles shift heavily when:  
- Camera is too low or too high  
- The arm is rotated toward the camera  

Solution:  
- Normalize keypoints using torso length  
- Switch to 3D pose with MediaPipe 3D landmarks  

### 3. Noise in Wrist/Elbow Landmark  
Solution:  
- Time-series smoothing  
- Angle-based rep detection rather than raw coordinate movement  

---

# ⭐ Future Improvements  
- Switch to 3D angle computation  
- Add ML model for classification of Good/Wrong reps  
- Add multi-person pose estimation  
- Add sound feedback for uploaded videos  
- Add a full dashboard with Streamlit charts  

---

# 🙌 Credits  
Built using **Python**, **MediaPipe**, **OpenCV**, and **Streamlit**.  
Designed for Smartan.AI Internship Assessment 2025.

