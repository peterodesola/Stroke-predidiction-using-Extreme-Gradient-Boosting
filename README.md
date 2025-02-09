# STROKE PREDICTION USING MACHINE LEARNING

## TABLE OF CONTENTS
- [INTRODUCTION](#INTRODUCTION)
- [DATA SCIENCE](#DATA-SCIENCE)
- [BENEFITS OF DATA SCIENCE](#BENEFITS-OF-DATA-SCIENCE)
- [THE BIG DATA](#THE-BIG-DATA)
- [CHALLENGES OF BIG DATA](#CHALLENGES-OF-BIG-DATA)
- [STROKE](#STROKE)
- [LITERATURE REVIEW](#LITERATURE-REVIEW)
- [STATISTICAL ANALYSIS OF STROKE](#STATISTICAL-ANALYSIS-OF-STROKE)
- [RACE AND ETHNICITY STATISTICS](#RACE-AND-ETHNICITY-STATISTICS)
- [AGE STATISTICS](#AGE-STATISTICS)
- [AIM AND OBJECTIVES](#AIM-AND-OBJECTIVES)
- [STATEMENT OF PROBLEM](#STATEMENT-OF-PROBLEM)
- [METHODOLOGY](#METHODOLOGY)
- [EXPLORATORY DATA ANALYSIS](#EXPLORATORY-DATA-ANALYSIS)
- [S/NO:	FEATURES:	DESCRIPTION:	TYPES OF VARIABLES](#S/NO:-FEATURES:-DESCRIPTION:-TYPES-OF-VARIABLES)
- [DATA PRE-PROCESSING](#DATA-PRE-PROCESSING)
- [APPLICATION OF MACHINE LEARNING ALGORITHM FOR PREDICTION](#APPLICATION-OF-MACHINE-LEARNING-ALGORITHM-FOR-PREDICTION)
- [MODEL SELECTION AND APPLICATION](#MODEL-SELECTION-AND-APPLICATION)
- [XGBOOST PERFORMANCE AND PROJECT CHALLENGES](#XGBOOST-PERFORMANCE-AND-PROJECT-CHALLENGES)
- [LIMITATIONS AND CHALLENGES](#LIMITATIONS-AND-CHALLENGES)
- [RECOMMENDATIONS](#RECOMMENDATIONS)
- [CONCLUSION](#CONCLUSION)


## INTRODUCTION
## DATA SCIENCE
Data science can be defined as the process by which data informatics, statistics, algorithms, data analysis and other related methods are being unified in order to analyse, interpret and understand data (Hayashi, 1998).
Data science encompasses of several fields like statistics, scientific methods, data analysis and artificial intelligence in extraction of quality values from data. Data science is a subset of artificial intelligence.
## BENEFITS OF DATA SCIENCE
Data science has become the top trending skills in every sector, particularly the health sector and we all need its application for quality analysis and interpretations of data. It has become so important in the health sector and really growing fast. It’s being beneficial in the following ways:
-	Data science application has made it easy in detecting the symptoms of disease at very early stage.
-	Doctors can now monitor patient’s health remotely with the vast development of technologies and tools through data science.
-	Data science has provided a deep understanding of genetic related disorder
-	The pharmaceutical companies have been using the application of data science to manufacture drugs.
-	Data science is being widely used for medical imaging, examples are MRI, X-Ray, Ultra-sound.
## THE BIG DATA
Big data according to McKinsey can be referred to as the datasets with sizes that are large and cannot be capture, store, maintain and analyse by typical database software tools (Manyika, 2011). The big data are also referred to as the 5V’s.
## THE CHALLENGES OF THE BIG DATA 
-	Capturing of clean, complete, accurate and well formatted data has been a major challenge.
-	Less storage for dataset.
-	Provision of maximum security for the devices from being hacked or prevention from malware.
## STROKE
The bleeding or blood clot in the brain which can cause permanent damage with great effect on mobility, cognition, sight or communication is as a result of stroke.
Stroke is an urgent medical state that leads to long term neurological destruction and can as well lead to death.
They can be classified into 2.
1.	The Ischemic embolic 
2.	Haemorrhagic
- The Ischemic embolic happens when a blood clot from the heart moves through the blood stream to the narrower brain arteries.
- The Haemorrhagic stroke happens as a result of ruptured arterial blood vessels or leakage of arterial blood vessel in the brain.
Some symptoms occur before a stroke happen which could be sudden numbness on one of the sides or part of the body, sudden confusion and difficulty in speech (Centres for Disease Control and Prevention, 2020).
Most times, other health-related issues surface before a person can have stroke but we don’t take cognisance of them. So therefore, it is essential to know and understand the risk factors behind stroke so that one can have appropriate and timely treatment to prevent it.
## LITERATURE REVIEW
To gain deeper insights into the "Predicting Stroke Using Machine Learning" dataset, I reviewed previous studies that utilized similar stroke datasets. This investigation allowed me to evaluate the performance of various models, ultimately guiding me in selecting the one most likely to yield optimal predictive results based on established research.
## STATISTICAL ANALYSIS OF STROKE
Cardiovascular disease was caused by stroke in 2020 which led to 1 in 6 deaths. Someone dies of stroke every 3.5 minutes and in the United States, someone is prompt to having stroke every 40 seconds.
In the United States, more than 795,000 people have a stroke every year, and ischemic strokes takes 87% of all strokes.
Long term disability has been caused by stroke (Tsao, 2022. Pg 145), it reduces survival movements especially the older ones from 65 years and above.
## RACE AND ETHNICITY STATISTICS
There has been reduction in stroke death rate since 2013, and blacks are twice at risk of stroke compared to Caucasians according to disease control centre.
## AGE STATISTICS
Stroke can happen any age, but risk increases with age. People less than 65 years
Old at rate of 38% were hospitalized in 2014. (Jackson, 2019.).
In 2017, Chantamit-o-pas, predicted stroke using deep learning, he predicted with the aid of neural network.
The models used for comparison in prediction accuracy were Naïve Bayes, SVM. Naïve Bayes was used for discrete predictor, while support vector machine was used for linear performance. The result shows that deep learning was good in heart stroke detection.
Another machine learning approach to stroke risk prediction project was carried out by Yu Sao et al. The prediction was carried out using machine learning models such as SVM, decision trees, nearest neighbours and multi-layer perception. SVM was highly recommended because of its evaluation metrics which was more accurate. It was stated that machine learning was much more perfect in prediction of stroke dataset with better accuracies. 
## AIM AND OBJECTIVES
The goal of this project is to forecast the likelihood of a stroke using key patient variables. To achieve this, we will:
- Leverage Data Mining: Apply data mining techniques to identify patients who are at risk of stroke.
- Analyze Key Variables: Examine various patient attributes to detect individuals who have a higher propensity for developing stroke.
- Develop a Predictive Model: Create and validate a machine learning model that accurately predicts stroke risk based on the identified factors.
## STATEMENT OF PROBLEM
Stroke remains one of the leading causes of disability and death worldwide, highlighting the need for effective early detection methods. Despite the availability of comprehensive stroke datasets, current approaches often fall short in accurately predicting stroke risk due to the complex interplay of various patient factors. This project addresses this gap by developing a machine learning model that utilizes critical features from a stroke dataset to predict the likelihood of stroke occurrence. The aim is to provide healthcare professionals with a reliable, data-driven tool for early identification of high-risk patients, ultimately improving preventative care and patient outcomes.
## METHODOLOGY
In this part, we will discuss the method used, including source of dataset, attributes, the data pre-processing algorithms and the metrics evaluation.
## EXPLORATORY DATA ANALYSIS
## DATA COLLECTION
The dataset was extracted from https://www.kaggle.com. It contains 5110 observations with 12 columns.
Our prediction is discrete output which shows that the project work is Classification machine learning because the output is either the individual might be at risk of having stroke or not.
Details of our dataset is shown below.
Details of dataset;5110 OBSERVATION, 12 COLUMNS
## S/NO:	FEATURES:	DESCRIPTION:	TYPES OF VARIABLES
1. GENDER:	MALE=2115, FEMALE=2994, OTHER=1.	 TYPE: CATEGORICAL/INDEPENDENT.
2. AGE:	THE YEARS ATTAINED BY INDIVIDUAL.	TYPE: INDEPENDENT/ORDINAL
3. HYPERTENSION:	IMPACT OF BLOOD PRESSURE ON STROKE.	 TYPE:INDEPENDENT/NOMINAL
4. HEART DISEASE:	IMPACT OF STATUS OF THE HEART CONDITION: 1=HAS HEART DISEASE, 0=NO HEART DISEASE.	 TYPE: INDEPENDENT/CATEGORICAL
5. EVER MARRIED:	EFFECT OF MARITAL STATUS ON STROKE PREDICTION: MARRIED= 1, NOT MARRIED=0	TYPE:INDEPENDENT/CATEGORICAL
6. WORK TYPE:	CHECKING IF THE NATURE OF JOB HAVE IMPACT ON TENDENCY OF HAVING STROKE: PRIVATE=0, SELF EMPLOYED=1, GOV JOB=2, CHILDREN=3, NEVER WORKED=4.	TYPE: INDEPENDENT/CATEGORICAL
7. RESIDENCE TYPE:	EFFECT OF INDIVIDUAL RESIDENCE ON STROKE: RURAL=0, URBAN=1. RURAL NOTE: RESIDENCE ARE BASICALLY PEOPLE LIVING IN VILLAGES OR NON- DEVELOPED ARE WHILE URBAN REFERS TO PEOPLE LIVING IN THE CITY.	    TYPE: INDEPENDENT/CATEGORICAL
8. AVG GLUCOSE LEVEL:	TO SEE THE EFFECT OF AVERAGE GLUCOSE LEVEL ON STROKE.	  TYPE: INDEPENDENT/NORMINAL
9. SMOKING STATUS:	EFFECT OF SMOKING STATUS ON POSSIBILITY OF HAVING STROKE: FORMERLY SMOKED= 0, NEVER SMOKE= 1, SMOKES= 2, UNKNOWN= 3	  TYPE: INDEPENDENT/CATEGORICAL
10. STROKE:	STROKE IS THE MAIN PROJECT WE ARE WORKING ON. IT IS OUR Y/OUTPUT VARIABLE. IT DEPENDS ON ALL OTHER VARIABLES FOR OUTPUT.	TYPE: DEPENDENT/CATEGORICAL
11. BMI:	BODY MASS INDEX EFFECT ON PREDICTION OF STROKE	     TYPE:INDEPENDENT/NOMINAL
## DATA PRE-PROCESSING
Data pre-processing is a fundamental data mining technique that transforms raw, unstructured data into a clean, consistent, and understandable format. Real-world datasets often contain errors due to inconsistencies, missing values, and incomplete trends. Pre-processing addresses these issues, ensuring that the data is reliable and ready for effective analysis and predictive modeling.
- Step 1: Import libraries necessary for the processing of our prediction

  ![Screenshot 2022-04-18 130718](https://github.com/user-attachments/assets/91c2c6a6-33a0-4c50-9e75-384032ac54c7)

- Step 2: Import the dataset

![Screenshot 2022-04-18 132329](https://github.com/user-attachments/assets/87ef2ce7-6740-4398-8d18-0f989798233c)


- Step 3: Check for the missing values. Since dataset could be messy and incomplete, so we need to perfect our data for quality outcome for prediction.
  
![Screenshot 2022-04-18 132959](https://github.com/user-attachments/assets/c59050dd-243f-4f1e-84af-0ea58ba56ebf)

We can fix the missing values using two methods.
1. If removing a row does not compromise our overall outcome, we can simply drop it from the dataset.  
2. Alternatively, missing values can be imputed by calculating statistical measures—such as the mean, median, or standard deviation—and using these values to fill in the gaps. The code snippet below demonstrates how to perform this imputation.
   
![Screenshot 2022-04-18 133542](https://github.com/user-attachments/assets/51b4b22d-f6c9-42e5-9f36-267aa87493ed)

![Screenshot 2022-04-18 135048](https://github.com/user-attachments/assets/78537ded-6e81-4cf7-af6e-5bc2d6a2b775)

- Step 4: Feature Scaling  
It is essential to encode categorical variables into numerical values. This conversion ensures that all features are in a consistent format, which is critical for accurate matching and visualization outcomes.

![Screenshot 2022-04-18 134258](https://github.com/user-attachments/assets/82c984c0-6f9e-49fd-a5e9-b9ea0759d61f)

Some values were also missing for gender when values were rechecked if there were any missing values.

![Screenshot 2022-04-18 135048](https://github.com/user-attachments/assets/7c4c3f60-81bb-4a1f-b32b-7f9328db4389)

ID feature won’t be needed in the prediction of stroke, so the ID feature was dropped from the column 

![Screenshot 2022-04-18 135310](https://github.com/user-attachments/assets/3dd64be4-276d-4972-b660-cffa8b348c86)

- Step 5: Visualization  
After addressing missing values and encoding categorical data into numerical values, we can proceed to visualize the key variables that contribute to stroke prediction. By comparing these features with the target variable (Stroke), we can uncover meaningful patterns and relationships. The visualizations below, along with their brief explanations, illustrate these comparisons and support our predictive analysis.

![Screenshot 2022-04-25 105846](https://github.com/user-attachments/assets/c23278bc-f596-4995-8102-f2ad6b9102ec)

A count plot visualization of work type and stroke incidence reveals that individuals in private employment exhibit the highest risk of stroke, followed by those who are self-employed. Government employees and children show similar stroke risk levels to the self-employed group, while individuals who have never worked display a comparatively lower risk.

![Screenshot 2022-04-25 124034](https://github.com/user-attachments/assets/d6cb0845-cb8f-461c-b9b6-84ad5f24e9e0)

![Screenshot 2022-04-25 124356](https://github.com/user-attachments/assets/82d7ccc3-020d-4f51-a36c-89eab14ee1ea)

![Screenshot 2022-04-25 134528](https://github.com/user-attachments/assets/a4346860-e174-4269-9f0b-601a4683de47)

![Screenshot 2022-04-25 134646](https://github.com/user-attachments/assets/01c48a44-d5ba-4078-a5d1-8e30a047e596)

![Screenshot 2022-04-25 141640](https://github.com/user-attachments/assets/6f7d7797-56c8-49e1-a344-5b1bf043f839)

![Screenshot 2022-04-25 141717](https://github.com/user-attachments/assets/c0b3071d-83e5-4ed6-b929-970e116db0a9)

## CALCULATING MINIMUM AND MAXIMUM AVERAGE GLUCOSE LEVEL
The obtained values of 55.12 and 271.74 reveal a significant disparity in the column's data, indicating the need for standardization. This large variation could negatively impact the accuracy of the predictions.

![Screenshot 2022-04-25 110842](https://github.com/user-attachments/assets/c34a38ec-f960-4d57-8b40-9b0e45902992)

 ## VARIABLE DISTRIBUTION
 
 ![Screenshot 2022-04-25 105846](https://github.com/user-attachments/assets/32bb5ea4-5564-4955-a5eb-d2d373de215b)

![Screenshot 2022-04-25 111244](https://github.com/user-attachments/assets/343a7b5d-601b-48c2-b619-bb2591631ab7)

![Screenshot 2022-04-18 143321](https://github.com/user-attachments/assets/20295912-4a93-4a2c-9606-3555f9286b88)

![Screenshot 2022-04-25 111930](https://github.com/user-attachments/assets/7ab540a0-ab2e-4dab-b509-26f9cec781fc)

![Screenshot 2022-04-25 114159](https://github.com/user-attachments/assets/07019cf0-dd56-47f6-af44-634e952bc54a)

![Screenshot 2022-04-25 114435](https://github.com/user-attachments/assets/c92eaf8d-a5aa-4bf6-823e-119170e98485)

In the visual display below, age increases the rate of stroke risk. Age 80 has the highest risk.

![Screenshot 2022-04-25 121403](https://github.com/user-attachments/assets/91c90530-d3b7-4863-8b0e-4c8e8f551855)

The BMI variable exhibits high positive skewness, indicating significant asymmetry in its distribution. This suggests that normalizing the dataset is necessary to improve the accuracy of our predictions.

![Screenshot 2022-04-25 121553](https://github.com/user-attachments/assets/b1a2efee-7bb8-46c7-8754-c2b1408e3923)

The average glucose level distribution exhibits a positive skew, and the dataset requires balancing to ensure accurate and reliable predictions.

![Screenshot 2022-04-25 122033](https://github.com/user-attachments/assets/690b5c3c-a879-41c7-acf4-99b602d9e59d)

Heatmaps provide a visual representation of data using color gradients, simplifying the understanding of complex datasets.  Correlation maps, specifically using Pearson's method, can be displayed as heatmaps to effectively visualize and analyze correlations within the data.

![Screenshot 2022-04-25 122933](https://github.com/user-attachments/assets/2dd4f4ce-eb75-4a4a-9692-f55e3cb4a3d8)

The visualization reveals correlations between several variables.  Specifically, "work type" and "BMI" exhibit a negative correlation, while "stroke" and "age" show a positive correlation.  Other variables also demonstrate varying degrees and directions of correlation.

![Screenshot 2022-04-25 123627](https://github.com/user-attachments/assets/218ae608-2e36-42a1-b5ea-c6647cf809be)

The scatterplot indicates a positive correlation between age and glucose level, suggesting that glucose levels tend to increase with age.
## APPLICATION OF MACHINE LEARNING ALGORITHM FOR PREDICTION
## Addressing Class Imbalance for Machine Learning Prediction
As observed in the visualizations, the target variable exhibits a significant class imbalance. This imbalance can negatively impact model performance, leading to biased predictions. To mitigate this issue, the Synthetic Minority Over-sampling Technique (SMOTE) will be employed. SMOTE addresses class imbalance by generating synthetic samples for the minority class.  The algorithm randomly selects a point from the minority class and identifies its K-nearest neighbors.  Synthetic samples are then created along the line segments joining the selected point and its neighbors. The following code demonstrates the implementation of SMOTE to balance the dataset.

![Screenshot 2022-04-18 135856](https://github.com/user-attachments/assets/063192f1-148e-4543-80e1-6463f66526ce)

The dataset is partitioned into training and testing sets to evaluate model performance and ensure generalization to unseen data. The feature matrix (independent variables) is split into X_train (training set) and X_test (testing set). Correspondingly, the dependent variable is divided into y_train (training labels) and y_test (testing labels). This separation allows the model to learn patterns from the training data (X_train, y_train) and then assess its predictive capabilities on the held-out test data (X_test, y_test). This process is crucial for assessing model accuracy and preventing overfitting. The code for splitting the data, including the specified percentages for the training and testing sets, is shown below.

![Screenshot 2022-04-18 140225](https://github.com/user-attachments/assets/d479b2c0-ff6c-40d2-b9a3-f02ff11f48cf)

## MODEL SELECTION AND APPLICATION
For stroke prediction, two classification models, Logistic Regression and XGBoost, were selected for comparative analysis to determine which provides higher accuracy and better predictive performance.
## LOGISTIC REGRESSION
Logistic Regression is a suitable model for predicting binary outcomes, where the dependent variable has two possible values (e.g., presence or absence of stroke). It estimates the probability of the outcome based on a set of independent variables.
## MODEL APPLICATION
![Screenshot 2022-04-18 140712](https://github.com/user-attachments/assets/fee25520-b3c6-40f8-b2b5-791fe26b2777)

The model achieved 83% accuracy, as measured by the F1-score.  Furthermore, the Area Under the Curve (AUC) score, exceeding 80%, suggests good predictive capability.
## XGBOOST
XGBoost is a versatile and powerful machine learning model suitable for both classification and regression tasks.  It is known for its effectiveness and often achieves high performance.
## Applying XGBoost

![Screenshot 2022-04-18 141149](https://github.com/user-attachments/assets/e0a3d150-3e61-4196-8bc0-f7aeb06b78d0)

## XGBOOST PERFORMANCE AND PROJECT CHALLENGES
XGBoost demonstrated superior performance, as evidenced by its F1-score and ROC curve.  Achieving 95% accuracy, XGBoost significantly outperformed Logistic Regression, making it the preferred model for this prediction task.
## MODEL EVALUATION
Model performance was evaluated using the F1-score. XGBoost achieved the highest accuracy, reaching 95%.
## LIMITATIONS AND CHALLENGES
This project encountered several challenges:
- Time Constraints: The limited timeframe restricted the extent of data exploration and model development.
- Missing Data: A significant amount of missing data in the BMI column posed a challenge, which was addressed using mean imputation.
- Class Imbalance: The imbalanced target variable necessitated the use of SMOTE to improve model performance.
- Tableau Connectivity and Display Issues: Difficulties connecting the dataset to Tableau and subsequent display problems, including unexpected characters and file generation, hindered data visualization efforts.
## RECOMMENDATIONS
For future stroke prediction projects, it is recommended to explore a wider range of models to optimize predictive accuracy. Longer project timelines are essential, and relying on multiple evaluation metrics is crucial for a comprehensive assessment of model performance.
## CONCLUSION
This stroke dataset prediction project involved several key steps: data preprocessing, feature selection (removing the ID column), handling missing values (mean imputation), standardization, and addressing the class imbalance using SMOTE.  XGBoost achieved the highest performance, with an F1-score of approximately 95%. Given the initial class imbalance, a confusion matrix was used to provide a more detailed understanding of the model's accuracy, further validating XGBoost's performance. Hyperparameter tuning was employed to optimize the XGBoost model and enhance its predictive capabilities. Data visualization was performed in both Python and Tableau, utilizing various dependent and independent variables to gain insights into the dataset and validate initial assumptions.


