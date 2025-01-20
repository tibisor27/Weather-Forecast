from django.shortcuts import render

# Create your views here.
import requests #luam date din API
import pandas as pd #manipularea datelor
import numpy as np #operatii numeice
from sklearn.model_selection import train_test_split #impartim datele pentr testare
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error 
from datetime import datetime, timedelta
import pytz
import os

API_KEY = 'eff1cd764cb3b08405273b6c3bb8e28a'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'


# import os

# file_path = '/Users/tibisor/Desktop/Proiect_PI/weatherProject/forecast/weather.csv'

# if os.path.exists(file_path):
#     print(f"Fisierul exista la calea: {file_path}")
# else:
#     print("Fisierul NU exista. Verifica locatia sau numele.")

# x = [10, 20, 30, 40]
# print(x)
# x = np.array(x) 
# print(x)
# x = x.reshape(-1, 1)   # Reshape în matrice cu o coloana
# print(x)

#FUNCTIE DE PRELUAREA A DATELOR
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    return {
        'city' : data['name'],
        'current_temp' : round(data['main']['temp']),
        'feels_like' : round(data['main']['feels_like']),
        'temp_min' : round(data['main']['temp_min']),
        'temp_max' : round(data['main']['temp_max']),
        'humidity' : round(data['main']['humidity']),
        'description' : data['weather'][0]['description'],
        'country' : data['sys']['country'],
        'wind_gust_dir' : data['wind']['deg'],
        'pressure' : data['main']['pressure'],
        'Wind_Gust_Speed' : data['wind']['speed'],
        'clouds' : data['clouds']['all'],
        'visibility' : round(data['visibility'] / 1000),
    }

#FUNCTIE DE CITIRE A DATELOR ISTORICE
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna() #sterge coloane care n au valori
    df = df.drop_duplicates()
    #print(df.head(5))
    return df

#PREGATIM DATELE PT TESTARE
def prepare_data(data):         #Funtia o sa transforme datele care nu sunt numerice in valori numerice 
    le = LabelEncoder()             #le este o variabila, o instanta a functie LabelEncoder(), functia asta transforma variabile non-numerice in variabile numerice.
    #print("Datele inainte de labelEncoder()", data['WindGustDir'])
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])  #'fit.transform' este o metoda a functiei care va transforma fiecare variabila din coloana "WindGustDir" din setul de date
                                                                 #'data' intr o variabila numerica(id). Deci se creeza un nou array(care va contine numere in loc de directii(S,W,N,E)) 
                                                                 #care va fi stocat in data['WindGustDir'] dupa ideea unui pointer.
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow']) #ACELASI MECANISM CA MAI SUS
    #print("Datele dupa labelEncoder()", data['WindGustDir'])

    x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']] #selectez mai multe coloane de.odata, datorita pandas
    y = data['RainTomorrow']
    return x,y,le

#TRAIN MODEL FUNCTION
def train_rain_model(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42) #impartim datele, 80% pt antrenament si 20% pt testare RandomForestClassifier este
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)               #un model de ML bazat pe arbori de decizie(100), cu "fit" se analizeaza relatia
    model.fit(x_train,y_train)                                                          #dintre date(x_train si y_train), x_train contine date caracteristice(vant,umiditate,presiune,etc)
                                                                                        #si y_train contine date de 0 si 1 specifice daca ploua sau nu. In felul asta
                                                                                        #'fit' analizeaza relatia dintre seturile astea de date pt a putea face viitoare predictii
    y_pred = model.predict(x_test)
    # print("Mean squared Error for Rain Model")

    # print(mean_squared_error(y_test,y_pred))
    return model

#PREPARE REGRESSION DATA
def prepare_regression_Data(data, feature):     #functia ia o coloana din date, si adauga fiecare valorea din liniile coloanei in listele x si y dupa modelul urmator.
    x, y = [], []

    for i in range(len(data) - 1):
        x.append(data[feature].iloc[i])         #asemanator cu 'data[feature][i]' dar e mai bine cu 'iloc' pt cazul cand indexului nu i nr ci o valoare.
        y.append(data[feature].iloc[i+1])

    x = np.array(x).reshape(-1,1)               #np.array(x) transforma lista x intr-o lista specifica numpy(pt ML) ex[2 4 5 6]; reshape transforma noua lista intr o matrice cu 1 coloana
                                                #'-1' semnifica ca numpy determina automat nr de randuri, iar '1' semnnifica nr de coloane
    y = np.array(y)                                 
    return x,y

#TRAIN REGRESSION MODEL
def train_regression_model(x,y):                #model ca cel de sus, de regresie(folosit pt a prezice valori discrete, ex: da sau nu )
    model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    model.fit(x,y)
    return model

#PREDICT FUTURE
def predict_future(model,current_value):
    predictions = [current_value]   #lista in care pastram predictiile

    for i in range(5):      
        next_value = model.predict(np.array([[predictions[-1]]])) #'predict returneaza o lista de predictii; modelul face predictie in functie de ultimul element din 'predictions'
                                                                  #acest ultim element este transformat intr un format specifit pentru model(lista np);deci next_value va fi mereu o lista
        predictions.append(next_value[0])                  #adauga valoarea predictiei(e doar una) in 'predictions' (folosim [0] pt ca next_value e lista chiar daca are 1 sg elem)

    return predictions[1:]                                 #returnam intreaga lista de predictii


#Weather Analysis
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        try:
            current_weather = get_current_weather(city)
        except KeyError:  # Dacă API-ul nu gaseste orasul
            error_message = f"City '{city}' not found"
            context = {
                'error_message': error_message,
                'location': city,  
            }
            return render(request, 'weather.html', context)
        #city = input("Enter any city name: ")
        current_weather = get_current_weather(city)
        #print(current_weather.keys())
        csv_path = os.path.join('/Users/tibisor/Desktop/Proiect_PI/weather.csv')
        date_istorice = read_historical_data(csv_path)

        x,y,le = prepare_data(date_istorice)  #x-contine toate coloane in afara de cea care zice daca ploua sau nu, y-contine coloana cu ploaia, le-instanta a fct LabelEncoder

        rain_model = train_rain_model(x,y)

        #map wind direction to campass points
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.25), ("E", 78.75, 101.25),("ESE",101.25, 123.75),
            ("SE", 123.75,146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.25), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WW", 281.25, 303.75), ("NW",303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)

        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_data = {
        'MinTemp' : current_weather['temp_min'], 
        'MaxTemp' : current_weather['temp_max'], 
        'WindGustDir' : compass_direction_encoded,
        'WindGustSpeed' : current_weather['Wind_Gust_Speed'],
        'Humidity' : current_weather['humidity'],
        'Pressure' : current_weather['pressure'],
        'Temp' : current_weather['current_temp'],
        }

        #print("datele inainte: ",current_data)
        current_df = pd.DataFrame([current_data])
        #print("datele actuale: ",current_df)

        rain_prediction = rain_model.predict(current_df)[0]

        x_temp,y_temp = prepare_regression_Data(date_istorice, 'Temp')

        x_hum, y_hum = prepare_regression_Data(date_istorice, 'Humidity')

        temp_model = train_regression_model(x_temp, y_temp)

        hum_model = train_regression_model(x_hum, y_hum)

        #predict future temperature and humidity

        future_temp = predict_future(temp_model, current_weather['temp_min'])

        future_humidity = predict_future(hum_model, current_weather['humidity'])

        #prepare time for future predictions

        timezone = pytz.timezone('Asia/Karachi')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours = 1)
        next_hour = next_hour.replace(minute = 0, second = 0, microsecond = 0)

        future_times = [(next_hour + timedelta(hours = i)).strftime("%H:00") for i in range(5)]

        # print(f"City: {city}, {current_weather['country']}")
        # print(f"Current Temperature: {current_weather['current_temp']}")
        # print(f"Feels like: {current_weather['feels_like']}")
        # print(f"Minimum Temperature: {current_weather['temp_min']} grade Celsius")
        # print(f"Maximum Temperature: {current_weather['temp_max']} grade Celsius")
        # print(f"Humidity: {current_weather['humidity']}%")
        # print(f"Weather Prediction: {current_weather['description']}")
        # print(f"Rain Prediction: {'Yes' if rain_prediction else 'No'}" )
        # print("\nFuture Temperature Predictions: ")

        # for time, temp in zip(future_times, future_temp):
        #     print(f"{time}: {round(temp,1)} grade Celsius")
        # print("\nFuture Humidity Predictions: ")

        # for time, humidity in zip(future_times, future_humidity):
        #     print(f"{time}: {round(humidity,1)}%")

    # weather_view()
        #stocam fiecare valoarea separat

        time1, time2, time3, time4, time5 = future_times
        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_humidity

        #dictionar pt valori
        context = {
            'location' : city,
            'current_temp' : current_weather['current_temp'],
            'MinTemp' : current_weather['temp_min'],
            'MaxTemp' : current_weather['temp_max'],
            'feels_like' : current_weather['feels_like'],
            'humidity' : current_weather['humidity'],
            'clouds' : current_weather['clouds'],
            'description' : current_weather['description'],
            'city' : current_weather['city'],
            'country' : current_weather['country'],

            'time' : datetime.now(),
            'date' : datetime.now().strftime("%B %d, %Y"),

            'wind' : current_weather['Wind_Gust_Speed'],
            'pressure' : current_weather['pressure'],

            'visibility' : current_weather['visibility'],

            'time1' : time1,
            'time2' : time2,
            'time3' : time3,
            'time4' : time4,
            'time5' : time5,

            'temp1' : f"{round(temp1, 1)}",
            'temp2' : f"{round(temp2, 1)}",
            'temp3' : f"{round(temp3, 1)}",
            'temp4' : f"{round(temp4, 1)}",
            'temp5' : f"{round(temp5, 1)}",

            'hum1' : f"{round(hum1, 1)}",
            'hum2' : f"{round(hum2, 1)}",
            'hum3' : f"{round(hum3, 1)}",
            'hum4' : f"{round(hum4, 1)}",
            'hum5' : f"{round(hum5, 1)}",

            }

        return render(request, 'weather.html', context)
    return render(request, 'weather.html')
