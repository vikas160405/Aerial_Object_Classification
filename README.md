# Aerial Object Classification — Bird vs Drone

GUVI × HCL Capstone Project. Classifies aerial images as **Bird** or **Drone** using
a Custom CNN and a MobileNetV2 transfer-learning model, deployed via Streamlit.

## Project Structure
```
├── aerial_classification_v2.ipynb   # Full pipeline: EDA, preprocessing, training, evaluation
├── app.py                           # Streamlit deployment app
├── requirements.txt
├── model_comparison_report_v2.csv   # Final metrics for both models
└── models/
    ├── best_model_v2.weights.h5     # Transfer MobileNetV2 (deployed model)
    ├── custom_cnn_v2.weights.h5     # Custom CNN
    └── best_model_info.txt          # Preprocessing metadata for app.py
```

## Results

| Model | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Custom CNN | 82.8% | 0.813 | 0.787 | 0.800 |
| **Transfer MobileNetV2 (deployed)** | **96.3%** | **0.957** | **0.957** | **0.957** |

Full confusion matrices and classification reports are in the notebook (Section 7).

## Running the App
```
pip install -r requirements.txt
streamlit run app.py
```
Upload an image; the app displays the predicted class and confidence score. The
"Auto-crop small/distant objects" option (on by default) improves accuracy on wide
shots where the subject is small — see Known Limitations below for what it can't fix.

## Project History: Why This Is a Rebuild

An earlier version of this project reported 99% test accuracy but was found, through
real-world testing, to misclassify photos where the bird or drone was small and
distant — sometimes confidently in the wrong direction. Debugging that report led to
three findings, all incorporated into this rebuild:

1. **The dataset is a repurposed object-detection export.** Every image is exactly
   416×416 with Roboflow-style filenames — the bird/drone was originally
   bounding-box-annotated and the classification version was built by cropping tightly
   to it. As a result, every training image shows the subject filling most of the
   frame; the model had never seen a genuinely small, distant subject. The high
   original test accuracy didn't catch this because the test set was drawn from the
   same biased source.

2. **The Custom CNN's poor performance (56%, barely above chance) was a training
   config problem, not an architecture problem.** It used a learning rate of 1e-4 —
   appropriate for fine-tuning a pretrained network, much too low for training a CNN
   from random initialization in a handful of epochs. Raising it to 1e-3 (with a
   temporary drop to 3e-4 partway through to stabilize a BatchNorm-related instability)
   brought it to 82.8% with the same architecture.

3. **Distance bias was fixed with synthetic augmentation.** Since the dataset has no
   real distant examples, ~50% of training images are now randomly shrunk and pasted
   onto a blurred version of themselves during training, teaching the model what a
   small/distant subject actually looks like. This measurably improved both models on
   a held-out synthetic "distant object" test set (MobileNetV2: 68.3% → 83.3%) and on
   a real photo of a distant bird flock that the original model got wrong.

## Known Limitations (Documented, Not Fixed — See Notebook Section 9 for Full Detail)

- **No "background/neither" class.** The model is strictly binary and will confidently
  misclassify objects that are neither a bird nor a drone (tested: a floating buoy was
  classified "Bird" at 99.9% confidence). Fixing this needs a 3rd class trained on
  negative examples, which this dataset doesn't include.
- **No trained object detector.** YOLOv8 (listed as optional in the brief) was not
  implemented — no bounding-box-annotated data was available for this dataset. The
  app's crop-assist feature is a lightweight edge-detection heuristic, not a trained
  detector, and can occasionally focus on the wrong object in cluttered scenes.
- **Synthetic, not real, distance data.** The augmentation approximates distance by
  shrinking existing close-up images; it is not a substitute for genuinely new
  photographs of distant birds/drones.

## Skills Demonstrated (per project brief)
Deep Learning, Computer Vision, Image Classification, Python, TensorFlow/Keras, Data
Preprocessing & Augmentation, Model Evaluation (accuracy/precision/recall/F1, confusion
matrix), Streamlit Deployment. Object detection (YOLOv8) was scoped out — see Known
Limitations.
