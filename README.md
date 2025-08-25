# CSE151A-Project
Classification Model for CSE151A Summer Session 2. 

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

Next steps to improve classification score will be testing our model with SVC and Decision Trees 

### Current Link to Juypter Notebook (Milestone 3) [CSE151A_Project](https://github.com/c4ngo/CSE151A-Project/blob/Milestone3/CSE151A_Project.ipynb)
