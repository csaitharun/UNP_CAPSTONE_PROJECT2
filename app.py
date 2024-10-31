import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request, render_template, jsonify
from joblib import load
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset and model
df = pd.read_csv("CustomerChurn.csv")
df_features = df.drop(columns=["Churn"])  # Drop the Churn column as it's the target
model = load("knn_model.joblib")

# Initialize StandardScaler
scaler = StandardScaler()
scaler.fit(df_features.select_dtypes(include=['float64', 'int64']))

@app.route("/", methods=['GET', 'POST'])
def loadPage():
    if request.method == 'POST':
        try:
            # Collect input data from the form
            input_data = [
                request.form[f'query{i}'] for i in range(1, 14)
            ]

            # Print input data for debugging
            print("Raw input data:", input_data)

            # Convert input data to the appropriate data types
            data = [[float(x) if i not in [8, 9, 10] else int(x) for i, x in enumerate(input_data)]]
            new_df = pd.DataFrame(data, columns=[
                'Call  Failure', 'Complains', 'Subscription  Length', 'Charge  Amount',
                'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
                'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age',
                'Customer Value'
            ])

            # Debugging: Print the DataFrame to ensure it has the correct data
            print("Input DataFrame after type conversion:", new_df)

            # Preprocess new data for prediction
            # Ensure that the DataFrame has the correct dtypes before scaling
            numerical_columns = new_df.select_dtypes(include=['float64', 'int64']).columns

            # Check if numerical columns exist and have valid data
            if not numerical_columns.empty:
                new_df_scaled = scaler.transform(new_df[numerical_columns])
                new_df.loc[:, numerical_columns] = new_df_scaled
            else:
                return jsonify({'error': 'No valid numerical data found for scaling.'})

            new_df_dummies = pd.get_dummies(new_df)
            new_df_dummies = new_df_dummies.reindex(columns=df_features.columns, fill_value=0)

            # Make predictions using the KNeighborsClassifier model
            single = model.predict(new_df_dummies)
            probability = model.predict_proba(new_df_dummies)

            # Generate output messages
            if single == 1:
                o1 = "This customer is likely to churn!!"
            else:
                o1 = "This customer is likely to continue!!"

            o2 = "Confidence: {:.2f}%".format(max(probability[0]) * 100)

            return render_template('home.html', output1=o1, output2=o2, 
                                   query1=request.form['query1'], 
                                   query2=request.form['query2'],
                                   query3=request.form['query3'],
                                   query4=request.form['query4'],
                                   query5=request.form['query5'], 
                                   query6=request.form['query6'], 
                                   query7=request.form['query7'], 
                                   query8=request.form['query8'], 
                                   query9=request.form['query9'], 
                                   query10=request.form['query10'], 
                                   query11=request.form['query11'], 
                                   query12=request.form['query12'], 
                                   query13=request.form['query13'])

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('home.html', query="")

if __name__ == "__main__":
    app.run(debug=True)
