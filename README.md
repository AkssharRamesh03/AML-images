# Adversarial Attacks on Object Detection Using FGSM

This project explores how adversarial examples can fool object detection models by applying the **Fast Gradient Sign Method (FGSM)** to a trained **Faster R-CNN** detector. The model is trained on a COCO-formatted dataset from Roboflow and tested by generating perturbed images that maintain visual similarity while altering predictions.

---

## Overview

Machine learning models—especially deep neural networks—are known to be vulnerable to small, intentional perturbations known as **adversarial attacks**. These changes are often invisible to the human eye but can drastically alter the model's output.

In this notebook:
- A Faster R-CNN model is trained for object detection on a custom dataset.
- The FGSM technique is applied to generate adversarial images.
- The model’s predictions are compared before and after the attack.
- The original image, noise, and adversarial image are visualized side by side.

---

## Dataset

The dataset used is hosted on [Roboflow](https://roboflow.com) and automatically downloaded using the Roboflow API.

- **Project**: `hepsi-humo1`
- **Workspace**: `school-ny3mt`
- **Version**: 2
- **Format**: COCO JSON
- **Number of Classes**: 15

Images and annotations are stored locally in the `./hepsi-2` directory after downloading.

---

## Features

- 📦 **Custom Dataset Loader**: Loads COCO-formatted annotations via `pycocotools`.
- 🏷️ **Model Customization**: Faster R-CNN with a modified classification head to fit 15 classes.
- ⚡ **FGSM Attack**: Fast implementation that perturbs a single test image to deceive the model.
- 📊 **Visualization**: Side-by-side plots of the original, noise, and adversarial image.
- 🔁 **Flexible Training**: Toggle between training from scratch or loading a pre-trained model.

---

## Project Structure

```plaintext
.
├── hepsi-2/ (will be downloaded while executing the code cells)
│   ├── train/
│   ├── test/
│   └── _annotations.coco.json
├── fgsm_object_detection.ipynb
├── README.md
└── cnn_model.pth  (generated after training)
```
## Running the Notebook

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AML-images.git
   cd AML-images```

2. **Install dependencies**
```pip install torch torchvision matplotlib pycocotools roboflow```

3. **Run the notebook**
   ```jupyter notebook fgsm_cav_object_detection.ipynb``


