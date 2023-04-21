from flask import Flask,render_template,url_for,request, send_file
import pandas as pd 
import spacy
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

df= pd.read_csv("Youtube04-Eminem.csv")
df_data = df[["CONTENT","CLASS"]]

df_x = df_data['CONTENT']
df_y = df_data.CLASS
data_strings = df_x
cv = CountVectorizer()
X = cv.fit_transform(data_strings) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.20, random_state=42)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train,y_train)
naive_bayes.score(X_test,y_test)

pickle.dump(naive_bayes, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)

@app.route('/predictfile',methods=['POST'])
def predictfile():
	if request.method == 'POST':
		uploaded_file = request.files['file']

		if not uploaded_file: 
			return render_template('home.html', error_message="Please Upload File")
		
		try:
			df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
		except: 
			return render_template('home.html', error_message="Please upload csv file")
		
		en = spacy.load('en_core_web_sm')

		sw_spacy = en.Defaults.stop_words
		print(df)
		df['clean'] = df['CONTENT'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (sw_spacy)]))
		count = cv.transform(df['clean'])

		predictions = model.predict(count)
		df['Prediction'] = predictions
		df.to_csv(df.to_csv('predictions.csv', index=False))
        
		return send_file('predictions.csv', as_attachment=True)
	return render_template('home.html')
if __name__ == '__main__':
	app.run(debug=True)