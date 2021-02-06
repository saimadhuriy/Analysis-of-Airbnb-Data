# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27, 2020

@author: Sai Madhuri Yerramsetti
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime 

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix

# Function to get difference between last review date and current date and return the days
def date_calc(date_calc):    
    dt = date_calc.date()
    cur_dt  = datetime.date.today()    
    lst_review_days = cur_dt - dt    
    last_review_days = lst_review_days.days    
    return last_review_days
    
if __name__ == "__main__":
    
    filelocation = "D:/Madhuri/Data Mining/Assignment2/seattle/listings.csv"
    # Reading listing CSV file using pandas
    raw_data = pd.read_csv(filelocation)
    # Below print statement will print first 5 records from listings csv file
    print(raw_data.head())
    # assigning pandas dataframe to the raw data
    data_frame = pd.DataFrame(data=raw_data)
        
    """
    DataFrame.info -- 
    1. This method will prints information about a DataFrame including 
    the index dtype and columndtypes, non-null values and memory usage
    """
    print(data_frame.info(verbose=True))
        
    """
    DataFrame.describe --
    2. This method will generate descriptive statistics. It include those 
    that summarize the central tendency , dipresion and shape of 
    dataset's distribution, excluding NaN(internally denote missing
    data)
    """
        
    print("Descriptive Statistics:")
    print(data_frame.describe())
        
    # Check the dimensionality of data
    print("No.of Rows :" +str(data_frame.shape[0]))
    print("No.of Cloumns :" +str(data_frame.shape[1]))
        
    # Select required columns        
    sel_df_col = data_frame[['id', 'host_id','host_since',
                             'host_is_superhost',
                             'host_total_listings_count','neighbourhood',
                             'property_type', 'room_type','accommodates',
                             'bedrooms','beds','price', 'security_deposit', 
                             'extra_people','minimum_nights',
                             'availability_30','availability_60',
                             'availability_90', 'availability_365',
                             'number_of_reviews', 'first_review', 
                             'last_review','review_scores_rating', 
                             'review_scores_value','cancellation_policy', 
                             'reviews_per_month']]

    # Check the dimensionality of data
    print("selected No.of Rows :" +str(sel_df_col.shape[0]))
    print("selected No.of Cloumns :" +str(sel_df_col.shape[1]))
        
    # Inspect the individual counts of each category in the categorical variables selected
    print("Host is Super Host Categorical counts:" +"\n" 
            +str(sel_df_col['host_is_superhost'].value_counts(dropna=False)))
        
    print("Host total listing Categorical Counts:" +"\n" 
            +str(sel_df_col['host_total_listings_count'].value_counts(dropna=False)))
        
    print("property type Categorical counts:" +"\n" 
            +str(sel_df_col['property_type'].value_counts(dropna=False)))
        
    print("room type Categorical counts:" +"\n" 
            +str(sel_df_col['room_type'].value_counts(dropna=False)))
        
    print("Accommodates Categorical counts:" +"\n" 
            +str(sel_df_col['accommodates'].value_counts(dropna=False)))
        
    print("Bedrooms Categorical counts:" +"\n" 
            +str(sel_df_col['bedrooms'].value_counts(dropna=False)))
    
    print("Beds Categorical counts:" +"\n" 
            +str(sel_df_col['beds'].value_counts(dropna=False)))
        
    print("Minimum Nights Categorical Counts :" +"\n"
            +str(sel_df_col['minimum_nights'].value_counts(dropna=False)))
        
    print("Cancellation Policy Categorical Counts : " + "\n"
            +str(sel_df_col['cancellation_policy'].value_counts(dropna=False)))
        
    print("Neighbourhood Categorical Counts :" + "\n"
            +str(sel_df_col['neighbourhood'].value_counts(dropna=False)))

    # select few columns to do further analysys
    listings_df = data_frame[['host_is_superhost',
                        'property_type','accommodates',
                        'bathrooms','bedrooms','beds',
                        'price','minimum_nights',
                        'availability_365','number_of_reviews',
                        'first_review','last_review',
                        'review_scores_rating','cancellation_policy', 
                        'reviews_per_month']]
    
    # print the variance between the columns
    print(listings_df.var())
    
    # print the correlation
    print(listings_df.corr())
    
    # get correlation matrix for listing columns selected
    covariant_matrix = listings_df.cov()
    
    # inspect the covariant matrix to check the relations between variables
    print(covariant_matrix)
    
    #inspect the summary statistics and column data types
    listings_df.describe()
    listings_df.info()
    
    # Check what are the columns with null values
    listings_df.loc[:, listings_df.isnull().any()]
    listings_df.loc[:, listings_df.notnull().all()]
    
    # inspect categorical value counts for the new dataframe
    print(listings_df['host_is_superhost'].value_counts(dropna=False))
    print(listings_df['property_type'].value_counts(dropna=False))
    print(listings_df['bedrooms'].value_counts(dropna=False))
    
    # Strip the dollar symbol from the 'price' colmun
    listings_df['price'] = listings_df['price'].apply(lambda x: x.lstrip('$'))
    
    # Remove the comma in price column
    listings_df['price'] = listings_df['price'].apply(lambda x: x.replace(",", ""))
    
    # Type cast the data tyoe of price column to int
    listings_df['price'] = pd.to_numeric(listings_df['price']).astype("float").astype("int")
    
    # Check the data types of all columns
    print(listings_df.dtypes)
    
    # Plot a heat using seaborn
    plt.figure(figsize=(15,10))
    sns.heatmap(listings_df.corr(), annot=True, linewidths=0.30, cmap='RdYlGn')
    plt.show()
    
    # draw a pair plot using seaborn
    sns.pairplot(listings_df)
    plt.show()
    
    # set the style of the graph to white and plot a lmplot with labelled axes and title
    sns.set(style="white")
    graph_4 = sns.lmplot(x="price", y="reviews_per_month", hue="host_is_superhost",
               height=5, data=listings_df)
    graph_4.set_axis_labels("Price", "Reviews per month")
    plt.title("Price vs number of reviews per month")
    
    # set the style of the graph to ggplot and plot a relplot with labelled axes and title
    plt.style.use('ggplot')
    graph_5 = sns.relplot(x="price", y="accommodates", hue="cancellation_policy",
            alpha=.7, palette="muted",
            data=listings_df)
    graph_5.set_axis_labels("Price", "Number of people accommodated")
    plt.title("Price vs number of people accommodated")
    
    # set the style of the graph to white and plot a jointplot with labelled axes and title
    sns.set(style="white")
    graph_6 = sns.jointplot(x="price", y="bathrooms",data=listings_df, xlim=(0, 500), ylim=(0, 12),
                  color="g")
    graph_6.set_axis_labels("Price", "Number of bathrooms")
    plt.title("Price vs number of bathrooms")
    
    # set the style of the graph to ggplot and plot a jointplot with labelled axes and title
    plt.style.use('ggplot')
    sns.jointplot(x='availability_365', y='price', data=listings_df)
    plt.xlabel("Per year Availability of the property")
    plt.ylabel("Price")
    plt.title("Price vs Per year Availability of the property")
    plt.show()

    # set the style of the graph to ggplot and plot a violin plot with stripplot with labelled axes and title
    plt.figure(figsize=(15,10))
    sns.violinplot(x='bedrooms', y='price', data=listings_df, inner=None, color='white')
    sns.stripplot(x='bedrooms', y='price', data=listings_df, size=4, jitter=True)
    plt.xlabel("Number of bedrooms")
    plt.ylabel("Price")
    plt.title("Price vs Number of bedrooms")
    plt.show()

    # set the style of the graph to white and plot a boxplot with labelled axes and title
    sns.set(style="white")
    plt.figure(figsize=(15,10))
    sns.boxplot(x="accommodates", y="price",
            hue="host_is_superhost", palette=["r", "b"],
            data=listings_df)
    plt.xlabel("Number of people accommodated")
    plt.ylabel("Price")
    plt.title("Price vs number of people accommodated")
    plt.show()
    
    #Convert the last_review to datetime object
    listings_df['last_review'] = pd.to_datetime(listings_df['last_review'])
    
    # Drop all rows with nan value in last_review column
    listings_df = listings_df.dropna(how="all", subset=['last_review'])
    
    # apply the date_calc() function to get the number of days and save it in a new column
    listings_df['last_review_days'] = pd.to_numeric(listings_df['last_review'].apply(date_calc))
    
    print("New Dates column:")
    print(listings_df['last_review_days'])
    
    # Check the category counts to check the unique value counts
    print(listings_df['bedrooms'].value_counts(dropna=False))
    print(listings_df['review_scores_rating'].value_counts(dropna=False))
    
    # drop the rows with nans in the specified columns
    listings_df = listings_df.dropna(how='any', subset=['bedrooms', 'review_scores_rating', 'bathrooms', 'beds'])
    
    # check the number the null values in the columns
    listings_df.isnull().sum()

    # Bin the price values into 3 equal bins
    price_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    
    # Stored the binned values in a new column
    listings_df['price_binned'] = price_discretizer.fit_transform(listings_df[['price']])
    
    # Bin the availability values into 10 equal bins
    availability_discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    
    # Stored the binned values in a new column
    listings_df['availability_binned'] = availability_discretizer.fit_transform(listings_df[['availability_365']])
    
    # Get the counts of the binned prices
    print("Binned price Categorical Counts :" + "\n"
            +str(listings_df['price_binned'].value_counts(dropna=False)))

    # draw a histogram of the new binned price
    listings_df.hist('price_binned')
    # Show the plot
    plt.show()

    # Get the counts of the binned availability
    print("Binned availability Categorical Counts :" + "\n"
            +str(listings_df['availability_binned'].value_counts(dropna=False)))

    # draw a histogram of the new binned availability
    listings_df.hist('availability_binned')
    # Show the plot
    plt.show()
    
    # Take the required columns for classification in a dataframe
    final_df = listings_df[['price_binned','accommodates','bathrooms','bedrooms','beds','availability_binned','number_of_reviews','reviews_per_month', 'last_review_days']]
    
    # Create variables X which contains all the columns that are used for classification and y that contain target variable
    y = final_df['price_binned'].values
    X = final_df.drop('price_binned', axis=1).values   

    ############################# KNN ################################

    # Scale the data first before applying Knn classifier
    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)

    # set the hyper parameters
    param_grid = {'n_neighbors' : np.arange(1, 50)}
    knn = KNeighborsClassifier()

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Instantiate the GridSearchCV object
    knn_clf = GridSearchCV(knn, param_grid, cv=10)
    
    # fit the model
    knn_clf.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = knn_clf.predict(X_test)

    # Compute and print metrics
    print("Accuracy of Knn classifier: {}".format(knn_clf.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Compute and print metrics
    print('Best parameters of Knn with Scaling: {}'.format(knn_clf.best_params_))
    print('Best score with Scaling: {}'.format(knn_clf.best_score_))
    
    # Plot the confusion matrix after normalizing it
    fig1 = plot_confusion_matrix(knn_clf, X_test, y_test, 
                            cmap=plt.cm.Blues, normalize = 'true')
    fig1.ax_.set_title("Confusion Matrix plot")
    print(fig1.confusion_matrix)
    plt.show()
    

    ########################## Decision Tree #############################
    
    # Setup the parameters and distributions to sample from: param_dist
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}

    # Instantiate a Decision Tree classifier: tree
    tree = DecisionTreeClassifier()

    # Instantiate the RandomizedSearchCV object: tree_cv
    tree_clf = RandomizedSearchCV(tree, param_dist, cv=10)

    # Fit it to the training data
    tree_clf.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = tree_clf.predict(X_test)
 
    # Compute and print metrics
    print("Accuracy of decision tree classifier: {}".format(tree_clf.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Print the tuned parameters and score
    print("Tuned Decision Tree Parameters: {}".format(tree_clf.best_params_))
    print("Best score is {}".format(tree_clf.best_score_))
    
    # Plot the confusion matrix
    fig2 = plot_confusion_matrix(tree_clf, X_test, y_test, 
                            cmap=plt.cm.Blues, normalize = 'true')
    fig2.ax_.set_title("Confusion Matrix plot")
    print(fig2.confusion_matrix)
    plt.show()
    
    ###################### Random Forest Classifier ####################
    
    # Setup the parameters and distributions to sample from: param_dist
    param_dist = {"max_depth": [3, None],
                "max_features": randint(1, 9),
                "min_samples_leaf": randint(1, 9),
                "n_estimators": randint(1, 100),
                "criterion": ["gini", "entropy"]}

    # Instantiate a Random forest classifier: tree
    random = RandomForestClassifier()

    # Instantiate the RandomizedSearchCV object: tree_cv
    random_clf = RandomizedSearchCV(random, param_dist, cv=10)

    # Fit it to the training data
    random_clf.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = random_clf.predict(X_test)
 
    # Compute and print metrics
    print("Accuracy of random forest classifier: {}".format(random_clf.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Print the tuned parameters and score
    print("Tuned random forest Parameters: {}".format(random_clf.best_params_))
    print("Best score is {}".format(tree_clf.best_score_))
    
    # Plot the confusion matrix
    fig3 = plot_confusion_matrix(random_clf, X_test, y_test, 
                            cmap=plt.cm.Blues, normalize = 'true')
    fig3.ax_.set_title("Confusion Matrix plot")
    print(fig3.confusion_matrix)
    plt.show()
    
    ######################### Logistic Regresion ################################# 
                            
    # Instantiate the logistic regression classifier: logreg with 10 fold cv
    logreg_clf = LogisticRegressionCV(cv = 10)

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

    # Fit the classifier to the training data
    logreg_clf.fit(X_train, y_train)

    # Predict the labels of the test data: y_pred
    y_pred = logreg_clf.predict(X_test)
    
    # Compute and print metrics
    print("Accuracy of Logistic regression classifier: {}".format(logreg_clf.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Plot the confusion matrix
    fig = plot_confusion_matrix(logreg_clf, X_test, y_test, 
                            cmap=plt.cm.Blues, normalize = 'true')
    fig.ax_.set_title("Confusion Matrix plot")
    print(fig.confusion_matrix)
    plt.show()
