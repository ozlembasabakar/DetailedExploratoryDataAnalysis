# Detailed Exploratory Data Analysis

## This project has 3 main parts. These are **Normal Distribution**, **Correlation** and **Heteroscedasticity**. First of all columns with a NULL data count of less than 0.3 were removed. Then the analysis were made.
---
### **Normal Distribution**: 
Numerical features were selected, and the data was analyzed to see if it fit the normal distribution. The methods listed below were used.

- ***Graphical Methods***: Plotted all the features. Their distribution was examined. Features that have similar distributions with the dependent variable (SalePrice) were found.

- ***Skewness and Kurtosis***: All features' skewness and kurtosis results were interpreted. 
 
- ***Statistical Methods***: For all features, test statistics values ​​were found and hypothesis tests were performed and interpreted.
    - ***Jarque-Bera***
    - ***Kolmogorov-Smirnov***


### **Correlation**: 
The correlation coefficients between all features and the dependent variable were found by removing outliers. Features with a positive correlation were kept in the golden_features list.


### **Heteroscedasticity and Removing**: 

- ***Breusch-Pagan***: It was examined with plots, how the features were distributed. Then heteroscedasticity was examined by applying the Breusch Pagan test.
- ***Removing***: Two methods are used to fix the           heteroscedasticity problem in features. 
    
    These are:
    - #### *Log Transform*
    - #### *Box Cox Transform*