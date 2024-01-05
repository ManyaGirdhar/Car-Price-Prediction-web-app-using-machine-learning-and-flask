from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/analysis')
def analysis():
    return render_template('eda.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')


fuel_code  = {

    0: 'diesel',
    1:'gas'
}


make_code = {
    0: 'alfa-romero',
    1: 'audi',
    2: 'bmw',
    3: 'chevrolet',
    4: 'dodge',
    5: 'honda',
    6: 'isuzu',
    7: 'jaguar',
    8: 'mazda',
    9: 'mercedes-benz',
    10: 'mercury',
    11: 'mitsubishi',
    12: 'nissan',
    13: 'peugot',
    14: 'plymouth',
    15: 'porsche',
    16: 'renault',
    17: 'saab',
    18: 'subaru',
    19: 'toyota',
    20: 'volkswagen',
    21: 'volvo'
}


# Define the columns
columns = ['length', 'width', 'curb-weight', 'engine-size', 'city-mpg',
       'highway-mpg', 'make', 'fuel-type']

pickle_file_path = 'model.pkl'
# Open the file in binary read mode and load the data
with open(pickle_file_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {}
        for column in columns:
            user_input[column] = float(request.form[column])

        # Create a DataFrame with the user input
        input_data = pd.DataFrame([user_input], columns=columns)

        print(model)
        price = model.predict(input_data )[0]
        
        input_data['fuel-type']= fuel_code[int( input_data['fuel-type'])]
        input_data['make']= make_code[int( input_data['make'])]
        
        # Display the input data
        return render_template('result.html', input_data=input_data.to_html(index=False), price=price)

    return render_template('index.html', columns=columns)

if __name__ == '__main__':
    app.run(debug=True)
