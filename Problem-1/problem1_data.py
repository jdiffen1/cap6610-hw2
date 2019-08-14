import numpy as np
#from numpy import genfromtxt
import csv

# Function for converting month abbreviations to numeric representations
def month_convert(month):
    abbr = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    return (abbr.index(month) + 1)

# ---- Convert nominal data to numeric data ---- #
# Open read file
pRead = open('bank.csv', 'rt')

# Open write file
fWrite = open('bank_feature.csv', 'w')
tWrite = open('bank_target.csv', 'w')

# 
read_file = csv.DictReader(pRead, delimiter=';')

# Set fieldnames for 'bank_feature.csv'
fnames = ['age', 
          'admin.', 'blue-collar', 'entrepreneur', 'housemaid',
          'management', 'retired', 'self-employed', 'services', 
          'student', 'technician', 'unemployed', 'job_unknown',
          'divorced', 'married', 'single', 'marital_unknown',
          'primary', 'secondary', 'tertiary', 'educ_unknown',
          'default_no', 'default_yes', 'default_unknown',
          'balance',
          'housing_no', 'housing_yes', 'housing_unknown',
          'loan_no', 'loan_yes', 'loan_unknown',
          'cellular', 'telephone', 'contact_unknown',
          'day',
          'month',
          #'jan', 'feb', 'march', 'apr', 'may', 'june', 'july',
          #'aug', 'sept', 'oct', 'nov', 'dec',
          'duration',
          'campaign',
          'pdays',
          'previous',
          'success', 'failure', 'other', 'poutcome_unknown']

# Set fieldnames for 'bank_target.csv'
tnames = ['y']

#
feature_file = csv.DictWriter(fWrite, delimiter=";", fieldnames = fnames)
target_file = csv.DictWriter(tWrite, delimiter=";", fieldnames = tnames)

# Write the header of the feature and target file
feature_file.writeheader()
target_file.writeheader()

# Convert all rows from bank.csv to numeric data and write to write file
for row in read_file:
    feature_file.writerow({'age': row['age'], 
                         'admin.': int(row['job'] == 'admin.'),
                         'blue-collar': int(row['job'] == 'blue-collar'), 
                         'entrepreneur': int(row['job'] == 'entrepreneur'), 
                         'housemaid': int(row['job'] == 'housemaid'),
                         'management': int(row['job'] == 'management'), 
                         'retired': int(row['job'] == 'retired'), 
                         'self-employed': int(row['job'] == 'self-employed'), 
                         'services': int(row['job'] == 'services'), 
                         'student': int(row['job'] == 'student'), 
                         'technician': int(row['job'] == 'technician'), 
                         'unemployed': int(row['job'] == 'unemployed'), 
                         'job_unknown': int(row['job'] == 'unknown'),
                         'divorced': int(row['marital'] == 'divorced'), 
                         'married': int(row['marital'] == 'married'), 
                         'single': int(row['marital'] == 'single'), 
                         'marital_unknown': int(row['marital'] == 'unknown'),
                         'primary': int(row['education'] == 'primary'), 
                         'secondary': int(row['education'] == 'secondary'), 
                         'tertiary': int(row['education'] == 'tertiary'), 
                         'educ_unknown': int(row['education'] == 'unknown'),
                         'default_no': int(row['default'] == 'no'), 
                         'default_yes': int(row['default'] == 'yes'), 
                         'default_unknown': int(row['default'] == 'unknown'),
                         'balance': row['balance'],
                         'housing_no': int(row['housing'] == 'no'), 
                         'housing_yes': int(row['housing'] == 'yes'), 
                         'housing_unknown': int(row['housing'] == 'unknown'),
                         'loan_no': int(row['loan'] == 'no'), 
                         'loan_yes': int(row['loan'] == 'yes'), 
                         'loan_unknown': int(row['loan'] == 'unknown'),
                         'cellular': int(row['contact'] == 'cellular'), 
                         'telephone': int(row['contact'] == 'telephone'), 
                         'contact_unknown': int(row['contact'] == 'unknown'),
                         'day': row['day'],
                         'month': month_convert(row['month']),
                         'duration': row['duration'],
                         'campaign': row['campaign'],
                         'pdays': row['pdays'],
                         'previous': row['previous'],
                         'success': int(row['poutcome'] == 'success'), 
                         'failure': int(row['poutcome'] == 'failure'), 
                         'other': int(row['poutcome'] == 'other'), 
                         'poutcome_unknown': int(row['poutcome'] == 'unknown')
                         })

    target_file.writerow({'y': int(row['y'] == 'yes') - int(row['y'] == 'no')
                         })

# Total number of rows in 'bank_feature.csv' and 'bank_target.csv' including 
# header row is 4522

# Convert data file 'bank_numeric.csv' into two arrays: the first for
# training and the second for testing.

# The array train_data has 2234 rows and 45 columns. The final column is the 
# target converted to +1 for 'yes' and -1 for 'no'.
#train_data = np.genfromtxt('bank_numeric.csv', skip_header=1, 
#                            skip_footer=2260, delimiter=';')
#print(train_data[0])
#print(train_data[2233])

# The array test_data has 2233 rows and 45 columns. The final column is the 
# target converted to +1 for 'yes' and -1 for 'no'.
#test_data = np.genfromtxt('bank_numeric.csv', skip_header=2262, delimiter=';')
#print(test_data[0])
#print(test_data[2232])
#print(test_data[0,0:44])

# Initialize weights for perceptron
#w_0 = 0
#w = np.zeros(44)

#print(np.dot(w, test_data[0,0:44]))
