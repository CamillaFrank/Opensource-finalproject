# Opensource-finalproject

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
   - [Dataset Download](#dataset-download)
   - [Dataset Structure](#dataset-structure)
3. [Algorithm](#algorithm)
   - [Gradient Boosting Classifier](#gradient-boosting-classifier)
     - [How Gradient Boosting Works](#how-gradient-boosting-works)
   - [Model Training](#model-training)
   - [Classification Process](#classification-process)
4. [Hyperparameters](#hyperparameters)
5. [Files in the Directory](#files-in-the-directory)
6. [Instructions](#instructions)
   - [Configuration](#configuration)
   - [Operating Instructions](#operating-instructions)
7. [Copyright and Licensing](#copyright-and-licensing)
8. [Contact Information](#contact-information)


## Overview
This project involves the classification of brain tumor images using machine learning techniques. The code utilizes the scikit-learn and scikit-image libraries for data processing, model training, and evaluation. The primary goal is to build a tumor classification model and identify the type of tumor from given image data.

## Dataset
The training dataset consists of brain tumor images categorized into four classes: glioma_tumor, meningioma_tumor, no_tumor, and pituitary_tumor. The images are preprocessed, resized to 64x64 pixels, and converted to grayscale.

**Note:** The dataset `tumor_dataset` is gitignored, so it does not come with the code. It needs to be downloaded separately and placed in the project directory.

### Dataset Download
To obtain the dataset, download it from e-class.

### Dataset Structure
Ensure the dataset is structured as follows:
```
tumor_dataset/
├── glioma_tumor/
├── meningioma_tumor/
├── no_tumor/
└── pituitary_tumor/
```
This structure corresponds to the four tumor classes. The images within each class should be stored in their respective directories.


## Algorithm
The chosen algorithm for tumor classification is the Gradient Boosting Classifier from the scikit-learn library. 

### Gradient Boosting Classifier
Gradient Boosting is an ensemble learning technique that combines the predictions of several weak learners (typically decision trees) to create a strong predictive model. The key idea behind Gradient Boosting is to sequentially train new models to correct the errors of the previous ones. This process minimizes the overall prediction error, making it a powerful tool for both classification and regression tasks.

#### How Gradient Boosting Works
1. **Initial Model:** The process starts with an initial weak model, often a shallow decision tree, which makes predictions on the training data.
2. **Residuals Calculation:** The difference between the actual and predicted values (residuals) is computed.
3. **Weighted Learning:** A new weak model is trained to predict the residuals, and the predictions are weighted based on their contribution to reducing the error.
4. **Iteration:** Steps 2 and 3 are repeated iteratively, with each new model focusing on correcting the errors of the combined ensemble of previous models.
5. **Final Ensemble:** The final prediction is made by combining the predictions of all weak models, each weighted by its contribution to minimizing the error.

### Model Training
In this project, the Gradient Boosting Classifier (`GradientBoostingClassifier`) from scikit-learn is used. The model is trained on the feature vectors of the training dataset (`X_train` and `y_train`).

### Classification Process
The trained model is then used to predict the labels for the test dataset (`X_test`). The predictions are obtained using the `predict` method.

## Hyperparameters
The primary hyperparameter is the random seed (`random_seed`) used for reproducibility. It is set to 42 in this implementation. Additionally, the model file is saved as "trained_model.joblib" for future use.

## Files in the Directory
- **main.ipynb:** The main Python script containing the code for data loading, preprocessing, model training, and evaluation.
- **tumor_dataset:** A directory containing the training dataset with subdirectories for each tumor class.
- **trained_model.joblib:** A pre-trained model file that can be used without rebuilding the model. Skip the model-building step by loading this file.

## Instructions
### Configuration
Ensure that the required packages are installed using the following command:
```bash
pip install scikit-learn scikit-image
```

### Operating Instructions
1. Execute the main script `main.ipynb`.
2. The script loads the dataset, preprocesses the images, and splits them into training and testing sets.
3. The Gradient Boosting Classifier is trained on the training set, and predictions are made on the test set.
4. Optionally, the model-building step can be skipped by using the pre-trained model file "trained_model.joblib."
5. The accuracy of the model on the test set is printed.

### Copyright and Licensing
This project is open-source and distributed under the MIT license. Refer to the LICENSE file for details.

### Contact Information
For questions or inquiries, please contact the project author at [cafra21@student.sdu.dk].


