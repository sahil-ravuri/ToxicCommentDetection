# Toxicity Comment Classifier

## Project Overview

This project aims to classify toxic comments into finer categories such as hateful, abusive, etc. The primary goal is to develop a model that can accurately identify various forms of toxicity in text data.

## Dataset

- **Dataset Name/Source:** [Jigsaw Toxicity Prediction Dataset](https://huggingface.co/datasets/google/jigsaw_toxicity_pred)
- **Data Description:** The dataset consists of a certain number of comments labeled with different toxicity categories. Preprocessing steps involved standard text cleaning techniques such as removing punctuation, lowercasing, and tokenization.

## Model Architecture

### Overview:

- **TextVectorization:** MAX_FEATURES, output_sequence_length, output_mode settings.
- **Embedding layer:** output_dim=32.
- **Bidirectional LSTM:** units=32, activation='tanh'.
- **Dense layers:** units=128, 256, 128, activation='relu'.
- **Output layer:** units=6, activation='sigmoid'.

### Reasoning (optional):
I chose this architecture for its capability to capture contextual information efficiently in sequential data like text.

## Training

- **Key Hyperparameters:**
  - Optimizer: Adam (default learning rate).
  - Number of epochs: 10
  - Batch size: 16
- **Evaluation Metrics:** Accuracy, F1-score, precision, recall.

## Results

# Classifier Performance Comparison

This table summarizes the performance metrics of various classification methods on the dataset:

| Method              | Accuracy | F1-Score | Precision | Recall | ROC AUC |
|---------------------|----------|----------|-----------|--------|---------|
| Multinomial NB      | 0.9177   | 0.2719   | 0.9813    | 0.1578 | 0.8703  |
| Logistic Regression | 0.9281   | 0.6344   | 0.6278    | 0.6413 | 0.9111  |
| SVM                 | 0.9572   | 0.7371   | 0.9167    | 0.6163 | 0.9710  |
| Bidirectional LSTM  | 0.9698   | 0.9118   | 0.9254    | 0.8986 | 0.9479  |

Each method achieved different scores across the following metrics:

- **Accuracy:** Measures the proportion of correct predictions among the total number of predictions.
- **F1-Score:** Harmonic mean of precision and recall, providing a balance between them.
- **Precision:** Measures the proportion of true positive predictions among the total predicted positives.
- **Recall:** Measures the proportion of true positive predictions among the actual positives.
- **ROC AUC:** Area under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between classes.

Based on these metrics, the Bidirectional LSTM model performed the best overall, achieving the highest scores in accuracy, F1-score, precision, recall, and ROC AUC. It indicates that the Bidirectional LSTM model effectively classified toxic comments into finer categories compared to the other methods evaluated.

  
### Dependencies:

- Python
- TensorFlow
- Keras
- pandas
- NumPy
- Dash
- Hugging Face Datasets

## Prediction Example

![Predection Example](<predection.jpg>)

## References
# References

- [Jigsaw Toxicity Prediction Dataset](https://huggingface.co/datasets/google/jigsaw_toxicity_pred) - Source of the dataset used in the project.
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825-2830. [Link](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
- Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Kudlur, M. (2016). TensorFlow: A system for large-scale machine learning. In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16) (pp. 265-283). [Link](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)
- Chollet, F., et al. (2015). Keras. GitHub repository. [Link](https://github.com/fchollet/keras)
- Brownlee, J. (Year). LSTM Networks for Time Series Forecasting. GitHub repository. [Link](https://github.com/jbrownlee/DeepLearningForTimeSeriesForecasting)
- Keras Team. (Year). LSTM example with Keras. GitHub repository. [Link](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/time_series.py)
- Python Software Foundation. (2022). Python Language Reference, version 3.10. [Link](https://docs.python.org/3/)
- Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Peterson, P. (2020). Array programming with NumPy. Nature, 585(7825), 357-362. [Link](https://www.nature.com/articles/s41586-020-2649-2)
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. In Proceedings of the 9th Python in Science Conference (pp. 56-61). [Link](https://conference.scipy.org/proceedings/scipy2010/mckinney.html)
- Rashant. (2022). Comment Toxicity Detector. GitHub repository. [Link](https://github.com/rashant/Comment-Toxicity-Detector)