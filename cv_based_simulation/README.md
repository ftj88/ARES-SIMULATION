# Computer Vision Rain & Fog Effects

This folder implements the **computer-vision–based simulation pipeline** that adds synthetic **rain** and **fog** effects to camera images.  
Currently, this version uses **dummy road images** for testing — the integration with the official IAC dataset will follow next.

---

## 1. Environment Setup

Clone the repository and navigate to the CV-based simulation folder:

git clone https://github.com/ftj88/ARES-SIMULATION.git
cd ARES-SIMULATION/cv_based_simulation

## 2. Create and activate a virtual environment:

python -m venv vrs_env
source vrs_env/Scripts/activate    # (Windows Git Bash or WSL)

## 3. Install dependencies:

pip install -r requirements.txt

## How to Run the Simulation:

From the cv_based_simulation root:
python -m scripts.apply_rainfog

This will:

- Read images from data/clean/camera/front/
- Apply rain and fog filters using OpenCV
- Save outputs to:
results/corrupted/light/camera/front/
results/corrupted/moderate/camera/front/
results/corrupted/heavy/camera/front/

## Notes
- The current version runs on sample images for demonstration.
- Integration with the IAC dataset and SLAM evaluation will come next.
- You can adjust the intensity parameters in effects/presets.py.