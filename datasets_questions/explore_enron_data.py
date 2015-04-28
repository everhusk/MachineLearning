
#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np
import pandas as pd

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#data = enron_data['PRENTICE JAMES']
#data = enron_data['COLWELL WESLEY']
#data = enron_data['LAY KENNETH L']
#data = enron_data['FASTOW ANDREW S']
#data = enron_data['SKILLING JEFFREY K']

payments = sum([item["total_payments"]=='NaN' for item in enron_data.values()])
percent = (float(payments)/len(enron_data)) * 100
print percent

pois = 0
count = 0
for v in enron_data.values():
    if v["poi"]:
        pois += 1
        if v["total_payments"] != 'NaN':
            count += 1

print payments
print pois
print count
print float(count)/pois * 100