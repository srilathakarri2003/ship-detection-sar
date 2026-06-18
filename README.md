# Ship Detection from Satellite Images using Deep Learning

A deep learning project that detects the presence of ships in satellite image patches using a Convolutional Neural Network (CNN), served through a Flask web application.

> Final-year mini project | K. Srilatha | Guide: Dr. T. Satish Kumar

## Overview

Maritime surveillance increasingly relies on automated analysis of satellite and SAR (Synthetic Aperture Radar) imagery, since these sensors can capture data in all weather conditions and at any time of day. This project builds a CNN-based binary classifier that labels image patches as **ship** or **no-ship**, and wraps it in a simple web interface for interactive testing.

### Existing approaches and motivation

Traditional ship detection pipelines segment sea and land regions using texture/shape cues, isolate a region of interest, propose candidate object locations (e.g. contrast-box or hierarchical segmentation techniques), and filter false positives in post-processing. Classical CNN architectures such as VGG-16 and VGG-19 have also been applied for feature extraction, often paired with classifiers like k-Nearest Neighbors.

This project takes a more direct deep learning approach: a compact CNN is trained end-to-end on labeled image patches, removing the need for hand-engineered segmentation and feature extraction steps.

## Architecture

```
ships dataset → preprocessing → split data → CNN model training
                                                     │
                                                     ▼
flask webapp → input image → trained model → predict ship/no-ship → shown on website
```

1. **Dataset collection** — labeled satellite image patches (ship / no-ship)
2. **Data analysis & preprocessing** — normalization, handling class imbalance, augmentation
3. **Training / validation / testing** — CNN trained on the prepared dataset
4. **Deployment** — trained model served via a Flask web app; users upload an image and get a real-time prediction

## Tech Stack

- **Python** — core language
- **TensorFlow / Keras** — CNN model definition and training
- **scikit-learn** — train/test splitting
- **Flask** — web application backend
- **HTML/CSS** — frontend upload interface

## Dataset

This project is set up to use the public Kaggle dataset **["Ships in Satellite Imagery"](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)** by Rhammell, which contains 80x80 RGB image chips labeled as ship (1) or no-ship (0).

To use it:
1. Download `shipsnet.json` from the Kaggle dataset page.
2. Place it inside the `data/` folder in this repo (create the folder if it doesn't exist).

## Project Structure

```
ship-detection-sar/
├── app.py                 # Flask web application
├── train_model.py         # CNN training script
├── requirements.txt       # Python dependencies
├── model/                 # Trained model saved here (ship_detector.h5)
├── templates/
│   └── index.html         # Web UI
├── static/
│   └── uploads/           # Uploaded images for prediction
├── data/                  # Dataset goes here (not included in repo)
└── notebooks/             # Optional: exploratory analysis / experiments
```

## Setup & Usage

```bash
# 1. Clone the repo
git clone https://github.com/srilathakarri2003/ship-detection-sar.git
cd ship-detection-sar

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Place shipsnet.json from Kaggle into the data/ folder

# 5. Train the model
python train_model.py --data data/shipsnet.json --epochs 15

# 6. Run the web app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser, upload a satellite image patch, and view the prediction.

## Model

A compact CNN with three convolution + max-pooling blocks, followed by a dense layer and a sigmoid output for binary classification (ship vs. no-ship). The architecture is intentionally lightweight so it trains quickly on CPU while still achieving strong accuracy on the shipsnet dataset.

## Future Improvements

- Extend to multi-class detection (ship type classification)
- Integrate SAR-specific preprocessing for all-weather imagery
- Add bounding-box object detection (e.g. YOLO) instead of patch-level classification
- Deploy the Flask app to a cloud platform (Render, Railway, or Hugging Face Spaces)

## Acknowledgements

- Dataset: [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) by Rhammell on Kaggle
