# Aerial Object Classification — Bird vs Drone
Classifies aerial images as **Bird** or **Drone** using
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
Upload an image, then drag the on-screen selection box to frame the bird or drone
you want classified — this is especially useful when the subject is small or far
away in the original photo. Only the selected region is passed to the model, which
then displays the predicted class and confidence score.

## Model Details

- **Preprocessing & Augmentation:** rotation, shift, shear, zoom, brightness,
  and horizontal flip, plus a synthetic distance-simulation augmentation
  (randomly shrinking the object and pasting it onto a blurred version of
  itself) so the model learns to recognize small/distant subjects, not just
  close-up ones.
- **Custom CNN:** 4 Conv+BatchNorm+MaxPool blocks trained from scratch at
  128×128 resolution.
- **Transfer Learning:** MobileNetV2 (frozen, ImageNet-pretrained base) with
  a custom classification head, trained at 224×224 resolution. This is the
  model used in the deployed app.
- **Training:** Adam optimizer, EarlyStopping and ModelCheckpoint on
  validation accuracy, with accuracy, precision, recall, and F1 tracked
  throughout.

## Known Limitations

- **No "background/neither" class.** The model is strictly binary and will
  confidently assign Bird or Drone even if the selected region is neither
  (e.g. a boat, buoy, or building). Resolving this would require training a
  3rd class on negative/background examples, which the current dataset does
  not include.
- **No trained object detector.** Object localization is handled by the user
  via manual crop selection rather than an automated detector (e.g. YOLOv8),
  since no bounding-box-annotated data was available for this dataset.
- **Synthetic, not real, distance data.** The distance augmentation
  approximates a far-away subject by shrinking existing close-up images; it
  is not a substitute for genuine photographs of distant birds/drones.

## Skills Demonstrated (per project brief)
Deep Learning, Computer Vision, Image Classification, Python, TensorFlow/Keras,
Data Preprocessing & Augmentation, Model Evaluation (accuracy/precision/recall/F1,
confusion matrix), Streamlit Deployment (including interactive image cropping).

