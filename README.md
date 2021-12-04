# Common-Functions
## 20-Nov-2021

This is a collection of common functions which are required while processing Machine Learning model. This will be continuous work on progress and I will keep updating the newer versions here.

File is still under testing.

It has following functions.

1. Load data: Loads data from XLSX and csv formats. Other formats to be added. Returns a Panda's dataframe
2. Prepare data: processes data for (a) NA values - Removes all NA values using 'any' (b) handles Nominal columns (c) Handles Ordinal data
3. Scale data - Scales data using StandardScaler or MinMaxScaler as selected by user
4. Test model - tests model and returns accuracy score, F1 score, RMSE values depending upon model used
5. Heat map - Generates heatmap of the confusion matrix
6. Get feature weights - Creates a Panda's dataframe with weights of all the features of a model. Index values are features and column values are weights
