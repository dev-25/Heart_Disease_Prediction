import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

heart_data = pd.read_csv("heart.csv")
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


def predict(input_data):

	input_data_as_numpy_array= np.asarray(input_data)

	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
	prediction = model.predict(input_data_reshaped)
	# print(prediction)	
	if (prediction[0]== 0):
	  st.subheader('The Person does not have a Heart Disease.')
	else:
	  st.subheader('The Person has Heart Disease.')	


def alg(input_data):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
	modell = RandomForestClassifier(random_state=0)
	modell.fit(X_train,y_train)

	input_data_numpy_array= np.asarray(input_data)

	input_data_reshaped = input_data_numpy_array.reshape(1,-1)
	predictionn = modell.predict(input_data_reshaped)

	if (predictionn[0]== 0):
	  st.subheader('The Person does not have a Heart Disease.')
	else:
	  st.subheader('The Person has Heart Disease.')	

def get_algo():

	st.header('Algorithm used in our project:')

	st.write('')
	st.write('')
	st.subheader('A. Logistic Regression:')

	st.write('')
	st.write('Logistic regression is used in various fields, including machine learning, most medical fields, and social sciences. Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.')

	st.write('')
	st.write('')
	st.write('The Accuracy of Logistic Regression is')
	# st.write(' = ', training_data_accuracy)

	st.write(format(training_data_accuracy,'.2f'))

	st.write('')
	st.write('')
	st.subheader('B. Random Forest Classifier')

	st.write('The Accuracy of Random Forest Classifier is')

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
	modelll = RandomForestClassifier(random_state=0)
	modelll.fit(X_train,y_train)
	X_train_predictt = modelll.predict(X_train)
	trainingddata_accuracy = accuracy_score(X_train_predictt, Y_train)
	
	st.write(format(trainingdata_accuracy,'.2f'))

def get_intro():
	st.header('Heart Disease Prediction.')
	st.write('')
	st.write('')

	st.subheader('The dataset contains the following information.')
	st.write('')
	st.write('1.  Age')
	st.write('2.  Sex')
	st.write('3.  Chest pain type (4 values)')
	st.write('4.  Resting blood pressure')
	st.write('5.  Serum cholestoral in mg/dl')
	st.write('6.  Resting electrocardiographic results (values 0,1,2)')
	st.write('7.  Maximum heart rate achieved')
	st.write('8.  Exercise induced angina')
	st.write('9.  Oldpeak = ST depression induced by exercise relative to rest')
	st.write('10.  The slope of the peak exercise ST segment')
	st.write('11.  Number of major vessels (0-3) colored by flourosopy')
	st.write('12.  thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

	st.write('')
	st.write('')
	st.image('data.png', caption='Dataset Preview')

	st.write('')
	st.write('')
	st.subheader('Dataset Link:')
	st.write('https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset')

def get_data():
	st.header('Enter Your Data 📝:')
	st.write('')
	st.write('')

	row1, row2 = st.columns((1,1))

	with row1:

		st.subheader('Age :')
		age = st.number_input('Enter your age in years', 0,150)	
		st.write('')

		st.subheader('Gender:')
		status = st.radio("Select your Gender", ('Male', 'Female'))
		if(status == 'Male'):
		 	gender=1
		else:
	   		gender=0
		st.write('')

		# gender = st.number_input('For male = 1, Female = 0', 0,1)

		st.subheader('Chest Pain:')
		chest_pain = st.number_input('Values between...(0,1,2,3)', 0,3)
		st.write('')


		st.subheader('Resting Blood Pressure:')
		restbps = st.number_input('Minimun = 80, Maximum = 200)', 80, 200)
		st.write('')

		st.subheader('Serum cholestoral in mg/dl:')
		chol = st.number_input('Minimun = 120, Maximum = 600', 120, 600)
		st.write('')
		
	st.subheader('Is fasting Blood Pressure > 120 mg/dl?')
		# fbs = st.number_input('Yes = 1, No = 0.', 0,1)
	row11, row22 = st.columns((1,1))
	with row11:
		statu = st.radio("", ('Yes', 'No'))
		if(statu == 'Yes'):
		 	fbs=1
		else:
	   		fbs=0
		st.write('')

	st.subheader('Resting electrocardiographic results:')	
	row12, row22 = st.columns((1,1))
	with row12:
		rer = st.number_input('Values between...(0,1,2)',0,2)
		st.write('')

	st.subheader('Maximum Heart Rate achieved:')
	row121, row252 = st.columns((1,1))
	with row121:
		maxheart = st.number_input('Minimun = 70, Maximum = 210', 70, 210)
		st.write('')
	
	st.subheader('Exercise induced angina:')
	# exin = st.number_input('Yes = 1, No = 0', 0, 1)
	row123, row242 = st.columns((1,1))
	with row123:
		stat = st.radio(".", ('Yes', 'No'))
		if(stat == 'Yes'):
		 	exin=1
		else:
	   		exin=0
		st.write('')

	st.subheader('Oldpeak = ST depression induced by exercise relative to rest:')
	row193, row72 = st.columns((1,1))
	with row193:
		oldp = st.number_input('Minimun = 0, Maximum = 6.2', 0.0, 6.2, 0.0, 0.1)
		st.write('')

	st.subheader('The slope of the peak exercise ST segment:')
	row1234, row2424 = st.columns((1,1))
	with row1234:
		slope = st.number_input('Minimun = 0, Maximum = 2', 0, 2)
		st.write('')

	st.subheader('Number of major vessels colored by flourosopy:')
	row1235, row2425 = st.columns((1,1))
	with row1235:
		color = st.number_input('Minimun = 0, Maximum = 3', 0, 3)
		st.write('')

	st.subheader('1 = Normal; 2 = Fixed defect; 3 = Reversable defect')
	row12356, row24725 = st.columns((1,1))
	with row12356:
		nor = st.number_input('Minimun = 1, Maximum = 3', 1, 3)
		st.write('')

	array = [age,gender,chest_pain,restbps,chol,fbs,rer,maxheart,exin,oldp,slope,color,nor]

	st.write('')
	st.subheader('The Inputed data is')
	st.write(array)
	st.write('')
	st.write('')
	st.header('Check for your Heart Disease 🫀!')

	st.write('')
	st.write('Using LogisticRegression:')
	if st.button('Check'):
		predict(array)
	# predict(array)
	st.write('')
	st.write('Using RandomForestClassifier:')
	if st.button('check'):
		alg(array)


st.sidebar.subheader('Select')
aa = st.sidebar.selectbox(" ",['Dataset', 'Prediction',])
# aa = st.sidebar.selectbox(" ",['Dataset', 'Prediction', 'Algorithm'])

if aa=='Prediction':
	get_data()
elif aa=='Dataset':
	get_intro()
