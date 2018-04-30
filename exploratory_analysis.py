# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 01:50:21 2018

@author: micts
"""

import  pandas as pd

data = pd.read_json("sample_data.json", lines=True)

def Descriptives(data):
    """
    Returns: Some descriptive statistics like mean, max, sd for the ratings.
    """
    return(data.overall.describe())
    
def NumberOfProducts(data):
    """
    Returns: The number of unique products.
    """
    return(len(data.asin.unique()))    

def NumberOfReviewers(data):
    """
    Returns: The number of uniquer reviewers
    """
    return(len(data.reviewerID.unique()))    

def MaxMinProduct(data):
    """
    Returns: Product ID and its mean rating throught the years, sorted in descending order.
             We consider the products with at least 20 reviews.
    """
    products = data.asin.value_counts()[data.asin.value_counts() > 20].index
    reviews = data[data.asin.isin(products)]
    reviews_sorted = reviews.groupby('asin')['overall'].mean().sort_values(ascending=False)
    return(reviews_sorted)
    
def MeanNumberReviews(data):
    """
    Returns: 1) The average number of products and reviewers
             2) The median of products and reviewers.
    """
    mean_num_per_product = data.asin.value_counts().mean()
    mean_num_per_reviewer = data.reviewerID.value_counts().mean()
    print('Mean number of reviews per product: {} \n'
          'Mean number of reviews per reviewer: {}'.format(mean_num_per_product, mean_num_per_reviewer))
    median_num_per_product = data.asin.value_counts().median()
    median_num_per_reviewer = data.reviewerID.value_counts().median()
    print('Median number of reviews per product: {} \n'
          'Median number of reviews per reviewer: {}'.format(median_num_per_product, median_num_per_reviewer))
  
def MeanOverallPerYear(data):
    """
    Returns: Mean value of ratings per year (1997-2014).
    """
    data['Datetime'] = pd.to_datetime(data['unixReviewTime'], unit='s')    
    data['Year'] = pd.DatetimeIndex(data['Datetime']).year
    return(data.groupby('Year').overall.mean())
   
def MeanOverallPerMonth(data):
    """
    Returns: Mean value of ratings per month.
    """
    data['Month'] = pd.DatetimeIndex(data['Datetime']).month
    return(data.groupby('Month').overall.mean())
    
# drop created columns, if they exist    
if 'Datetime' in data.columns:
    data.drop('Datetime', axis=1)
elif 'Year' in data.columns:
    data.drop('Year', axis=1)    
elif 'Month' in data.columns:
    data.drop('Month', axis=1)    