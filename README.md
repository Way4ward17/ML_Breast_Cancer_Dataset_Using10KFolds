# ML_Breast_Cancer_Dataset_Using10KFolds



<p><strong>CHAPTER THREE</strong></p>
<p><strong>RESEARCH METHODOLOGY</strong></p>
<p><strong>3.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; INTRODUCTION</strong></p>
<p>Breast cancer the most common cancer among women worldwide accounting for 25 percent of all cancer cases and affected 2.1 million people in 2015 early diagnosis significantly increases the chances of survival. The key challenge in cancer detection is how to classify tumors into malignant or benign machine learning techniques can dramatically improves the accuracy of diagnosis. Research indicates that most experienced physicians can diagnose cancer with 79 percent accuracy while 91 percent correct diagnosis is achieved using machine learning techniques.</p>
<p>In this case study, our task is to classify tumors into malignant or benign tumors using features of pain from several cell image.</p>
<p><strong>3.2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; METHODOLOGY</strong></p>
<p>Methodology is the systematic, theoretical analysis of the methods applied to a field of study. It comprises the theoretical analysis of the body of methods and principles associated with a branch of knowledge. Typically, it encompasses concepts such as paradigm, theoretical model, phases and quantitative or qualitative techniques. A methodology does not set out to provide solutions it is therefore, not the same as a method. Instead, a methodology offers the theoretical underpinning for understanding which method, set of methods, or best practices can be applied to a specific case, for example, to calculate a specific result.</p>
<p>In other words, methodology is the general research strategy that outlines the way in which research is to be undertaken and, among other things, identifies the methods to be used in it. These methods, described in the methodology, define the means or modes of data collection or, sometimes, how a specific result is to be calculated. Methodology does not define specific methods, even though much attention is given to the nature and kinds of processes to be followed in a particular procedure or to attain an objective. When proper to a study of methodology, such processes constitute a constructive generic framework, and may therefore be broken down into sub-processes, combined, or their sequence changed. (Katsicas, 2009)</p>
<p>In this work, we have considered the use of support vector machine and artificial neural network. &ldquo;Support Vector Machine&rdquo; (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. However,&nbsp; it is mostly used in classification problems. On the other hand is neural networks. It is a very powerful algorithm when you have massive datasets. This means that the neural network has enough data to create statistical models of the data which has been inputted, this is why they have been becoming more and more successful because of the amount of new data coming out every year.</p>
<p>For implementation, the researcher will be using ANACODA NAVIGATOR V1.5 where we will be using Jupyter notebook environment as our editor. At the offline environment.</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p><strong>FIGURE 3.1&nbsp; METHODOLOGY WORKFLOW</strong></p>
<p><strong>3.2.1 DATASET COLLECTION </strong></p>
<p>The dataset used for this work was downloaded from kaggle. Kaggle is a online dataset repository own by google with numerous dataset. The dataset in in a CSV format with 32 columns which are described as follows.</p>
<p><strong>Id</strong>: ID number</p>
<p><strong>Diagnosis</strong>: The diagnosis of breast tissues (M = malignant, B = benign)</p>
<p><strong>Radius_mean</strong>: mean of distances from center to points on the perimeter</p>
<p><strong>Texture_mean</strong>: standard deviation of gray-scale values</p>
<p><strong>Perimeter_mean:</strong> mean size of the core tumor</p>
<p><strong>Area_mean:</strong> area of the tumor shape</p>
<p><strong>Smoothness_mean:</strong> mean of local variation in radius lengths</p>
<p><strong>Compactness_mean</strong>: mean of perimeter^2 / area - 1.0</p>
<p><strong>Concavity_mean</strong>: mean of severity of concave portions of the contour</p>
<p><strong>Concave points_mean</strong>: mean for number of concave portions of the contour</p>
<p>Symmetry_mean</p>
<p><strong>Fractal_dimension_mean</strong>: mean for "coastline approximation" - 1</p>
<p><strong>Radius_se</strong>: standard error for the mean of distances from center to points on the perimeter</p>
<p><strong>Texture_se</strong>: standard error for standard deviation of gray-scale values</p>
<p><strong>Perimeter_se</strong></p>
<p><strong>Area_se</strong></p>
<p><strong>Smoothness_se</strong>: standard error for local variation in radius lengths</p>
<p><strong>Compactness_se</strong>: standard error for perimeter^2 / area - 1.0</p>
<p><strong>Concavity_se:</strong> standard error for severity of concave portions of the contour</p>
<p><strong>Concave points_se</strong>: standard error for number of concave portions of the contour</p>
<p><strong>Symmetry_se</strong></p>
<p><strong>Fractal_dimension_se</strong>: standard error for "coastline approximation" - 1</p>
<p><strong>Radius_worst:</strong> "worst" or largest mean value for mean of distances from center to points on the perimeter</p>
<p><strong>Texture_worst:</strong> "worst" or largest mean value for standard deviation of gray-scale values</p>
<p><strong>Perimeter_worst</strong></p>
<p><strong>Area_worst</strong></p>
<p><strong>Smoothness_worst</strong>: "worst" or largest mean value for local variation in radius lengths</p>
<p><strong>Compactness_worst</strong>: "worst" or largest mean value for perimeter^2 / area - 1.0</p>
<p><strong>Concavity_worst</strong>: "worst" or largest mean value for severity of concave portions of the contour</p>
<p><strong>Concave points_worst</strong>: "worst" or largest mean value for number of concave portions of the contour</p>
<p><strong>Symmetry_worst</strong></p>
<p><strong>Fractal_dimension_worst</strong>: "worst" or largest mean value for "coastline approximation" &ndash; 1</p>
<p>&nbsp;</p>
<p><strong>&nbsp;</strong></p>
<p><strong>FIGURE 3.1&nbsp; SCREENSHOTS OF THE PROPOSED MODEL</strong></p>
<p><strong>&nbsp;</strong></p>
<p><strong>3.2.2&nbsp;&nbsp;&nbsp; DATA PREPARATION/PRE-PROCESSING</strong></p>
<p>Data preprocessing is a data mining technique which is used to transform the raw data in a useful and efficient format. Due to the repository the breast cancer dataset is downloaded from. It has already been converted from raw data to a csv format file, which would make it easy for the analysis. Converting raw data into useful format isnt the only thing done here, it also cleaning and filtering the dataset, which is to remove redundanct data or missing value rows. In other to clean the dataset, the researcher employ the use of an additional software to do so, which is Rapid Miner. Rapid miner is a data scientist tool for data mining and machine learning task. With rapid miner, we were able to remove the missing value. We then convert back to csv for further processes.</p>
<p><strong>3.2.3&nbsp;&nbsp;&nbsp; TRAINING MODEL</strong></p>
<p>Training of our model would be implemented using python programming language and Jupyter note book. Jypyter notebook is a text editor in anaconda. And Anaconda is a python environment with machine learning tools. We would be use more of sci-kit learn library. Sci-kit learn is a python machine learning library which is embedded with numerous machine learning algorithm. We would be using the Numpy and Pandas library as well.</p>
<p>The ski-kit learn library would be used to split the dataset gotten in the ratio of 70:30, training and test set respectively. It will also provide of the accuracy of our prediction. &nbsp;</p>
<p><strong>&nbsp;</strong></p>
<p><strong>3.2.4&nbsp;&nbsp;&nbsp; TESTING MODEL</strong></p>
<p>After the model has been trained and saved. We can now use the sci-kit learn predict function to make a new prediction. In this case, the data supplied would have an empty value in the class label.</p>

<br><br>
This is a machine learning project. Using 10KFold Cross Validation. The algorithm used are;<br>
Artificial Nueral Network<br>
Decision Tree<br>
Support Vector Machine<br>
Naive Bayes<br>
<img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/1.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/2.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/3.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/4.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/5.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/6.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/7.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/8.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/9.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/10.png"/><br>
<img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/11.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/12.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/13.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/14.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/15.png"/><br><img src="https://github.com/Way4ward17/ML_Breast_Cancer_Dataset_Using10KFolds/blob/master/16.png"/><br>
