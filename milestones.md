### MILESTONE 2 UPDATES:

### Link to dataset: [civil_comments](https://huggingface.co/datasets/google/civil_comments)

The setup environment for this project is the following:

- __Python 3.9__
- __numpy__
- __matplotlib__
- __seaborn__
- __scikit-learn__
- __datasets__
  
The virtual environment used is __Jupyter__

### Link to Jupyter Notebook (Milestone 2): [CSE151A_Project](https://github.com/c4ngo/CSE151A-Project/blob/Milestone2/CSE151A_Project.ipynb)

### MILESTONE 3 UPDATES:

Added major preprocessing data such as:
  - dropped majority of non-toxic comments to reduce skewnessed
  - added two new labels __toxic_comments__ and __non_toxic_comments__ with values 0 or 1

Added new feature which is a threshold on the _toxic_ attribute. If a data point surpasses a threshold, the data point will be grouped to toxic comments. 

We trained our first model using Multinomial Naive Bayes on an alpha of 0.1 with a feature threshold of 0.5. 
The results of this model based on the classification report was:

Training Accuracy 82%

Training F1 Score 82%

Testing Accuracy 80%

Testing F1 Score 81%

Conclusion:

Our toxic classifcation model performed relatively well given the dataset with alpha = 10.0 and threshold = 0.5. Our training and testing data both had accuracy and f1 score of above 80%. Changing the hyperparameters did not improve our 1st model significantly. In order to improve our model furthermore, we may consider SVMs or Decision Trees. 

Next steps to improve classification score will be testing our model with SVC and Decision Trees 

### Current Link to Juypter Notebook (Milestone 3) [CSE151A_Project](https://github.com/c4ngo/CSE151A-Project/blob/Milestone3/CSE151A_Project.ipynb)

### MILESTONE 4 UPDATES:

#### Second Model: Logistic Regression with SVD

For our second model, we implemented a pipeline combining TF-IDF for text vectorization, Truncated SVD for dimensionality reduction, and a Logistic Regression classifier. Hyperparameter tuning was performed using GridSearchCV to find the optimal number of SVD components and the best regularization parameter (C) for the classifier.

#### Fitting Graph Analysis

The Logistic Regression with SVD model demonstrates a **good fit**, positioning it in the optimal region of a fitting graph where both bias and variance are low.

*   **Low Bias**: The model achieved a high validation accuracy of 84.44% and an F1-score of 83.75%, a notable improvement over the first model's 80% accuracy. This indicates that the model is complex enough to capture the underlying patterns in the data effectively.
*   **Low Variance**: The performance on the training set (84.83% accuracy, 84.07% F1-score) is very close to the validation set performance. This small gap suggests that the model generalizes well to new, unseen data and is not overfitting.

This combination of low bias and low variance places the model in the "Good Balance" zone of the fitting graph.

#### Conclusion

The second model, a Logistic Regression classifier enhanced with Truncated SVD, proved to be highly effective for this text classification task. The grid search identified the optimal hyperparameters as `C=5` for logistic regression and `n_components=1000` for SVD, resulting in a peak cross-validated F1-score of **83.08%**.

The final model achieved a **validation accuracy of 84.44% and an F1-score of 83.75%**. An analysis of the confusion matrix reveals that the model correctly identified 8,000 toxic and 8,881 non-toxic comments, showing a strong ability to distinguish between the two classes.

**Potential Improvements:**
*   **Advanced Feature Engineering**: Using more sophisticated text representations like Word2Vec or GloVe embeddings could help the model better grasp semantic context and nuances in language.
*   **Exploring More Complex Models**: To potentially capture more complex, non-linear patterns, we could train advanced models like Support Vector Machines (SVM) or gradient-boosted trees (e.g., XGBoost).
*   **Training on More Data**: While trained on a balanced 100,000-comment dataset, increasing the size of this balanced set could further improve the model's robustness and generalization capabilities.

#### Predictions on the Validation Set

The following are the prediction results from the validation dataset, which contained 20,000 comments (10,000 toxic, 10,000 non-toxic):

*   **Correct Predictions: 16,881**
    *   **True Positives (Correctly identified as toxic)**: 8,000
    *   **True Negatives (Correctly identified as non-toxic)**: 8,881
*   **Incorrect Predictions: 3,119**
    *   **False Positives (Incorrectly identified as toxic)**: 1,119
    *   **False Negatives (Incorrectly identified as non-toxic)**: 2,000

#### Next Steps: Future Models

Based on the project's progress, the next logical models to explore are:

1.  **Support Vector Classifier (SVC)**: SVCs excel in high-dimensional feature spaces like the ones created by TF-IDF. They are powerful because they can find the optimal hyperplane that separates classes, allowing them to model complex decision boundaries effectively. This makes them a strong candidate for potentially outperforming Logistic Regression.

2.  **Decision Trees and Ensemble Models**: While individual decision trees can be prone to overfitting, they are the building blocks for highly effective ensemble methods like **Random Forests** or **Gradient Boosting Machines (e.g., XGBoost, LightGBM)**. These models are industry standards for classification tasks because they can capture non-linear relationships and feature interactions, often leading to state-of-the-art performance.
