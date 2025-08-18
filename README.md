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

### Link to Jupyter Notebook: [CSE151A_Project](https://github.com/c4ngo/CSE151A-Project/blob/Milestone2/CSE151A_Project.ipynb)

How will you preprocess your data? Handle data imbalance if needed. You should only explain (do not perform pre-processing as that is in MS3) this in your README.md file and link your Jupyter notebook to it

### Preprocessing data
Firstly, we will deal with null values and duplicates by removing them. 
We will clean up the text data by converting it to lower case, removing numbers, emojis, etc. as they do not help in training our model. 
Our dataset is also very imbalanced with most of the entries having a 0 on each label. Thus, we will train our model on a subset of the dataframe with a significantly smaller proportion of these entries so as not to bias our model to always predict 0 or towards 0. 
