from Flask import render_template, url_for, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=["POST", "GET"])
def predict():
    dataset = pd.read_csv('data/Diabetes.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    classifier.score(x_test, y_test)

    # Alternative Usage of Saved Model
    ytb_model = open("classifier_pickle.pkl", "rb")
    classifier = joblib.load(ytb_model)

    if request.method == 'POST':
        value = request.form['Pregnancies'], request.form['PlasmaGlucose'], request.form['DiastolicBloodPressure'], \
                request.form['TricepsThickness'], request.form['SerumInsulin'], request.form['BMI'], request.form[
                    'DiabetesPedigree'], request.form['Age']
        data = [value]
        vect = sc.transform(data)
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)
    comment = request.form['comment']
    data = [comment]
    vect = cv.transform(data).toarray()
    my_prediction = classifier.predict(vect)
return render_template('result.html')


if __name__ == '__main__':
app.run(debug=True)
