![Apache License](https://img.shields.io/hexpm/l/apa)  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)    ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)   ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![Made with matplotlib](https://user-images.githubusercontent.com/86251750/132984208-76ce70c7-816d-4f72-9c9f-90073a70310f.png)  ![seaborn](https://user-images.githubusercontent.com/86251750/132984253-32c04192-989f-4ebd-8c46-8ad1a194a492.png)  ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white) ![tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white) ![medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white) ![coursera](https://img.shields.io/badge/Coursera-0056D2?style=for-the-badge&logo=Coursera&logoColor=white) ![udemy](https://img.shields.io/badge/Udemy-EC5252?style=for-the-badge&logo=Udemy&logoColor=white)

<a href="url"><img src="https://user-images.githubusercontent.com/86251750/146022279-cd570526-da5b-48ec-b1ac-952d74bcb626.jpg" height="500" width="1500" ></a>

<a href='https://www.freepik.com/photos/business'>Business photo created by rawpixel.com - www.freepik.com</a>

## Human Resources

* Hiring and retaining employees are extremely complex tasks that require capital, time and skills.

* Small business owners spend 40% of their working hours on tasks that do not generate any income such as hiring.

* Companies spend 15%-20% of the employee's salary to recruit a new candidate.

* An average company loses anywhere between 1% and 2.5% of their total revenue on the time it takes to bring a new hire up to speed.

* Hiring a new employee costs an average of $7645 (0-500 corporation).

* It takes 52 days on average to fill a position.

[source](https://toggl.com/blog/cost-of-hiring-an-employee)

## Acknowledgements

 - [python for ML and Data science, udemy](https://www.udemy.com/course/python-for-machine-learning-data-science-masterclass)
 - [ML A-Z, udemy](https://www.udemy.com/course/machinelearning/)
 - [ML by Stanford University ](https://www.coursera.org/learn/machine-learning)

## Appendix

* [Aim](#aim)
* [Dataset used](#data)
* [Run Locally](#run)
* [Exploring the Data](#viz)
   - [Dashboard](#dashboard)
   - [Matplotlib](#matplotlib)
   - [Seaborn](#seaborn)
* [feature engineering](#fe)
* [prediction with various models](#models)
* [conclusion](#conclusion)

## AIM:<a name="aim"></a>

The HR team collected extensive data on their employees and with the help of this we develop a model that could *predict which employees are more likely to quit*. 

## Dataset Used:<a name="data"></a>

The team provided with an extensive data, here's a sample of the dataset: 

`JobInvolvement`

`Education`

`JobSatisfaction`

`PerformanceRating`

`RelationshipSatisfaction`

`WorkLifeBalance`

[source of dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## Run locally:<a name="run"></a>

Clone the project

```bash
  git clone https://github.com/pradeepsuyal/Human_Resources
```

Go to the project directory

```bash
  cd Human_Resources
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```

## Exploring the Data:<a name="viz"></a>

I have used pandas, matplotlib and seaborn visualization skills.

**Matplotlib:**<a name="matplotlib"></a>
--------
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.You can draw up all sorts of charts(such as Bar Graph, Pie Chart, Box Plot, Histogram. Subplots ,Scatter Plot and many more) and visualization using matplotlib.

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install matplotlib:

    pip: pip install matplotlib

    anaconda: conda install matplotlib
    
    import matplotlib.pyplot as plt

![matplotlib](https://eli.thegreenplace.net/images/2016/animline.gif)

for more information you can refer to [matplotlib](https://matplotlib.org/) official site

**Seaborn:**<a name="seaborn"></a>
------
Seaborn is built on top of Python’s core visualization library Matplotlib. Seaborn comes with some very important features that make it easy to use. Some of these features are:

**Visualizing univariate and bivariate data.**

**Fitting and visualizing linear regression models.**

**Plotting statistical time series data.**

**Seaborn works well with NumPy and Pandas data structures**

**Built-in themes for styling Matplotlib graphics**

**The knowledge of Matplotlib is recommended to tweak Seaborn’s default plots.**

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install seaborn:

    pip: pip install seaborn

    anaconda: conda install seaborn
    
    import seaborn as sns
    
![seaborn](https://i.stack.imgur.com/uzyHd.gif)

for more information you can refer to [seaborn](https://seaborn.pydata.org/) official site.

**Dashboard:**<a name="dashboard"></a>
------
![HR dept](https://user-images.githubusercontent.com/86251750/146025879-adff16d3-4238-44d9-adf1-4d681e411327.png)

dashboard created with [tableau](https://public.tableau.com/app/profile/pradeep7347/viz/HRdeptanalysis/HRdept_)

**Screenshots from notebook**

![download](https://user-images.githubusercontent.com/86251750/146030345-f4c983a1-b433-49b3-9a07-650ef2c644d7.png)

![download](https://user-images.githubusercontent.com/86251750/146030566-3ce71280-f945-4a6a-ab75-36a6ab87ba32.png)

    Here red color represents 'employees who have left' and blue color represents 'employee who stayed'
    
![download](https://user-images.githubusercontent.com/86251750/146030922-4e1f034b-5fe6-4ade-838a-2ed8864bca62.png)

![download](https://user-images.githubusercontent.com/86251750/146031006-d07e2c62-7063-4a72-96d5-42792527e3ab.png)

## My approaches on Feature Engineering<a name="fe"></a>
-------

* performed data cleaning.
* converted categorical features using OneHotEncoder.
* train-test split.
* performed scaling with MinMax scaler.
* Separating dependent and independent features.
* Training, Prediction and Evaluting using various models and then selected the best model as my final model.

You can read my blog on How to deal with categorical features at [medium](https://medium.com/analytics-vidhya/how-to-deal-with-categorical-features-for-machine-learning-17c6c160ea1) 


## Prediction with various Models:<a name="models"></a>
------

I have used various classification models for the prediction.

**LOGISTIC REGRESSION CLASSIFIER**

* Linear regression is used to predict outputs on a continuous spectrum. 

      Example: predicting revenue based on the outside air temperature. 

* Logistic regression is used to predict binary outputs with two possible values labeled "0" or "1"

      Logistic model output can be one of two classes: pass/fail, win/lose, healthy/sick
      
![image](https://user-images.githubusercontent.com/86251750/146035729-a43e75d7-765f-42d9-83d8-73f1e7f55516.png)

* Logistic regression algorithm works by implementing a linear equation first with independent predictors to predict a value. 

* We then need to convert this value into a probability that could range from 0 to 1.

      Linear equation:
      𝑦=𝑏_0+𝑏_1∗𝑥

      Apply Sigmoid function:
      𝑃(𝑥)= 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 (𝑦)
      𝑃(𝑥)=1/1+𝑒^(−𝑦)
      𝑃(𝑥)=1/1+𝑒^−(𝑏_0+𝑏_1∗𝑥)
      
![image](https://user-images.githubusercontent.com/86251750/146035395-b52a4449-cdbc-4230-b46e-07933f13b6b6.png)

* Now we need to convert from a probability to a class value which is “0” or “1”.

![image](https://user-images.githubusercontent.com/86251750/146036372-d6ed12d6-a414-450a-9ce2-bfdd1c46c2b4.png)

*evaluting logistic regression performance*

                precision    recall  f1-score   support

           0       0.91      0.98      0.94       312
           1       0.83      0.43      0.56        56

    accuracy                           0.90       368
    
![download](https://user-images.githubusercontent.com/86251750/146036994-39916f4d-da70-43e9-8108-c6f376c4d66c.png)


**RANDOM FOREST CLASSIFIER**
       
* Decision Trees are supervised Machine Learning technique where the data is split according to a certain condition/parameter. 

* Let’s assume we want to classify whether a customer could retire or not based on their savings and age.

* Random Forest Classifier is a type of ensemble algorithm. 

* It creates a set of decision trees from randomly selected subset of training set. 

* It then combines votes from different decision trees to decide the final class of the test object.

![image](https://user-images.githubusercontent.com/86251750/146037858-931d7e3c-c6ff-408a-b3fc-49e19715022c.png)

*evaluting RandomForest performance*

                precision    recall  f1-score   support

           0       0.86      0.98      0.92       310
           1       0.62      0.17      0.27        58

    accuracy                           0.85       368

![download](https://user-images.githubusercontent.com/86251750/146038383-e33c73c2-e7a0-41df-85c0-fa83ad6f6f61.png)

**DEEP LEARNING MODEL WITH TENSORFLOW**

![image](https://user-images.githubusercontent.com/86251750/146039065-781dbdfc-d38b-4f32-8f34-c8f468aa15f7.png)

*evaluting tensorflow performance*

                precision    recall  f1-score   support

           0       0.91      0.92      0.91       312
           1       0.51      0.46      0.49        56

    accuracy                           0.85       368

![download](https://user-images.githubusercontent.com/86251750/146039595-37940dc2-6136-4700-a939-4ee2c0c474cb.png)

**CONFUSION MATRIX**

![image](https://user-images.githubusercontent.com/86251750/146039996-e08e81d3-6f82-417f-b73a-b72e1fb4c97d.png)

* A confusion matrix is used to describe the performance of a classiﬁcation model: 

`True positives (TP): cases when classiﬁer predicted TRUE (they have the disease), and correct class was TRUE (patient has disease).`

`True negatives (TN): cases when model predicted FALSE (no disease), and correct class was FALSE (patient do not have disease).`

`False positives (FP) (Type I error): classiﬁer predicted TRUE, but correct class was FALSE (patient did not have disease).` 

`False negatives (FN) (Type II error): classiﬁer predicted FALSE (patient do not have disease), but they actually do have the disease`

`Classiﬁcation Accuracy = (TP+TN) / (TP + TN + FP + FN)` 

`Precision = TP/Total TRUE Predictions = TP/ (TP+FP) (When model predicted TRUE class, how often was it right?)` 

`Recall = TP/ Actual TRUE = TP/ (TP+FN) (when the class was actually TRUE, how often did the classiﬁer get it right?)`

## CONCLUSION:<a name="conclusion"></a>
-----

I have used various classification models for prediction but it looks that Logistic regression performs quite well so I have choosed it as my final model.

    NOTE--> we can further improve the performance by using other classification model such as Tree based models(XGBOOST, LightGBM, AdaBoost, CatBoost and many more) and further
            performance can be improved by using various hyperparameter optimization technique such as optuna, hyperpot, Grid Search, Randomized Search, etc.

