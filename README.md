# Scribble to Script

Handwritten text recognition: turn images of handwritten words into text. A CNN + LSTM network trained with CTC loss on the IAM handwriting dataset, built as a team of 4 (team lead).

**Best result:** 60.3% validation accuracy (run 5: batch size 25, 22 epochs). The full write-up is in [REPORT.md](REPORT.md); slides are in `Scribble To Script - PPT.pdf`.

## How it works

1. **Pre-process** – read word images in greyscale, apply thresholding, resize to a fixed shape
2. **Model** – convolutional layers extract features; bidirectional LSTM layers read them as a sequence of characters
3. **Loss** – CTC (Connectionist Temporal Classification) handles words of different lengths
4. **Export** – the best checkpoint by validation accuracy is saved as a `.tflite` model

## Experiments

Each notebook is one training run. File names read `Scribble_to_script_<run>_<batch size>_<epochs>.ipynb`.

| Run | Batch size | Epochs | Train accuracy | Validation accuracy |
|---|---|---|---|---|
| 1 | 10 | 10 | 37.5% | 36.5% |
| 2 | 20 | 20 | 93.3% | 58.5% |
| 3 | 30 | 20 | 81.1% | 53.1% |
| 4 | 30 | 8 | 28.6% | 30.3% |
| **5** | **25** | **22** | **94.6%** | **60.3%** |
| 6 | 21 | 15 | 75.7% | 53.0% (best 53.8%) |
| 7 | 18 | 13 | 66.5% | 25.1% (best 51.2%) |
| 8 | 10 | 12 | 48.8% | 45.0% |
| 9 | 2 | 10 | 48.7% | 43.6% |

Figures are the final-epoch values printed in each notebook. The gap between training and validation accuracy in the longer runs shows overfitting – the next thing to fix with augmentation or regularisation.

## How to run

The notebooks run in Google Colab. They expect the IAM words dataset as zip files in a Google Drive folder (`DATA_DIR` in the first cells); set that path to your own copy, then run all cells.

## Tech

Python · TensorFlow / Keras · OpenCV · NumPy · Google Colab
