from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import CustomerUserSerializer
from .models import CustomUser
import os
import google.generativeai as genai
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
import json
import pandas as pd
from pmdarima import auto_arima
import matplotlib
from operator import itemgetter
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a GUI
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from .models import Dimcars
from rest_framework import status
from PIL import Image
import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from rest_framework.decorators import api_view
import json
from rest_framework.decorators import api_view
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from django.db import connection
import numpy as np
from rest_framework.decorators import api_view
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.db import connection
from prophet import Prophet
from django.views.decorators.csrf import csrf_exempt
from django_pandas.io import read_frame
from sklearn.metrics import mean_absolute_error
import pandas as pd


@api_view(['GET'])
def apiOverview(request):
    api_urls = {
        'List': '/user-list/',
        'Create':'/user-create'
    }
    return Response(api_urls)
@api_view(['GET'])
def ShowAll(request):
    users=CustomUser.objects.all()
    serializer = CustomerUserSerializer(users,many=True)
    return Response(serializer.data)

@api_view(['POST'])
def CreateUser(request):
    if request.method == 'POST':
        serializer = CustomerUserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

@api_view(['GET'])
def ViewUser(request, pk):
    user = CustomUser.objects.filter(id=pk).first()
    if user is not None:
        serializer = CustomerUserSerializer(user)
        return Response(serializer.data, status=status.HTTP_200_OK)
    else:
        return Response({'error': 'L\'utilisateur spécifié n\'existe pas.'}, status=status.HTTP_404_NOT_FOUND)
    
@api_view(['POST'])
def updateUser(request, pk):
    users = CustomUser.objects.get(id=pk)
    serializer = CustomerUserSerializer(instance=users, data=request.data)
    if serializer.is_valid():
        serializer.save()

    return Response(serializer.data)


@api_view(['GET'])
def deleteUser(request, pk):
    user = CustomUser.objects.get(id=pk)
    user.delete()

    return Response('Items delete successfully!')
@api_view(['POST'])
def UserLogin(request):
    email = request.data.get('email')
    password = request.data.get('password')

    try:
        user = CustomUser.objects.get(email=email)
        if password == user.password:
            if user.role == 'leader':
               
                return Response({'message': 'Connexion réussie', 'role': user.role}, status=status.HTTP_200_OK)
            
            elif user.role == 'sales':
               
                return Response({'message': 'Connexion réussie', 'role': user.role}, status=status.HTTP_200_OK)
            elif user.role == 'operations':
               
                return Response({'message': 'Connexion réussie', 'role': user.role}, status=status.HTTP_200_OK)
            elif  user.role == 'marketing':
               
                return Response({'message': 'Connexion réussie', 'role': user.role}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Accès non autorisé'}, status=status.HTTP_403_FORBIDDEN)
        else:
            return Response({'error': 'Email ou mot de passe incorrect'}, status=status.HTTP_401_UNAUTHORIZED)
    except CustomUser.DoesNotExist:
        return Response({'error': 'Utilisateur non trouvé'}, status=status.HTTP_401_UNAUTHORIZED)
    


@api_view(['GET'])
def scoring(request):
    # Connect to your MySQL database
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        database='dw_abracadata'
    )

    # Define your SQL query to retrieve the necessary data
    query = """
        SELECT 
            dcl.client_name, 
            COUNT(fr.request_pk) AS reservation_count,
            MIN(da.annee) AS first_year,
            MAX(da.annee) AS last_year
        FROM 
            dimclients dcl
        JOIN 
            factrequest fr ON dcl.client_pk = fr.client_fk
        JOIN 
            dimdates da ON fr.arrivals_date_fk = da.date
        GROUP BY 
            dcl.client_name
    """

    # Retrieve data from MySQL database into Pandas DataFrame
    data = pd.read_sql(query, conn)

    # Close the database connection
    conn.close()

    # Store client names before dropping the column
    client_names = data['client_name']

    # Add a target variable indicating whether a client is a good candidate for promotion
    def determine_promotion(row):
        if row['reservation_count'] > 10 and (row['first_year'] == 2023 or (row['first_year'] == 2022 and row['last_year'] == 2023)):
            return 1
        else:
            return 0

    data['target'] = data.apply(determine_promotion, axis=1)

    # Define features (X) and target variable (y)
    X = data.drop(columns=['target', 'client_name'])
    y = data['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use LDA as a classifier
    lda_classifier = LinearDiscriminantAnalysis()
    lda_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = lda_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate discount percentage based on prediction
    discount_percentage = pd.Series(y_pred, index=X_test.index).map({1: 15, 0: 5})  # 15% for predicted good candidates, 5% for others

    # Create a dictionary to store the predictions data
    predictions_data = {
        'predictions': [
            {'client_name': client_name, 'prediction': prediction, 'discount_percentage': discount}
            for client_name, prediction, discount in zip(client_names[X_test.index], y_pred, discount_percentage)
        ]
    }
    predictions_data['predictions'] = sorted(predictions_data['predictions'], key=itemgetter('discount_percentage'), reverse=True)


    # Return the predictions data as JSON response
    return Response(predictions_data)


# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Setup database connection


class CarRecommendationAPIView(APIView):
    def post(self, request, format=None):
        engine = create_engine('mysql+mysqlconnector://root@localhost/dw_abracadata')
        req_type = request.data.get('req_type')
        trimester = request.data.get('trimester')

        if not req_type or not trimester:
            return Response({"error": "req_type and trimester are required fields."}, status=status.HTTP_400_BAD_REQUEST)

        # Define the SQL query
        query = """
        SELECT 
            rt.req_type,
            dc.car_type,
            dd.trimestre,
            COUNT(*) AS y  -- Count of occurrences
        FROM 
            factrequest fr
        JOIN dimrequesttypes rt ON fr.req_fk = rt.req_type_pk
        JOIN dimcars dc ON fr.car_fk = dc.car_pk
        JOIN dimdates dd ON fr.arrivals_date_fk = dd.date
        GROUP BY 
            rt.req_type, dc.car_type, dd.trimestre;
        """

        # Load the data into a DataFrame
        data = pd.read_sql_query(query, engine)

        # Encoding categorical variables
        le_req_type = LabelEncoder()
        data['req_type_encoded'] = le_req_type.fit_transform(data['req_type'])

        le_car_type = LabelEncoder()
        data['car_type_encoded'] = le_car_type.fit_transform(data['car_type'])

        le_trimester = LabelEncoder()
        data['trimester_encoded'] = le_trimester.fit_transform(data['trimestre'])

        # Define features and target
        X = data[['req_type_encoded', 'trimester_encoded']]
        y = data['car_type_encoded']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a RandomForest Classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        # Ensure that the inputs are in the format the model was trained on
        if req_type not in le_req_type.classes_ or trimester not in le_trimester.classes_:
            return Response({"error": "req_type or trimester not found in the label encoder classes."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Encode the inputs
        req_type_code = le_req_type.transform([req_type])[0]
        trimester_code = le_trimester.transform([trimester])[0]

        # Create input for predictions
        input_array = np.array([[req_type_code, trimester_code]])

        # Get the probability distribution across all car types
        probas = classifier.predict_proba(input_array)[0]

        # Get indices of the top 3 predictions
        top_indices = np.argsort(probas)[-3:][::-1]

        # Convert indices to car types
        top_cars = le_car_type.inverse_transform(top_indices)
        
        # Get the top 3 car types along with their respective probabilities
        top_cars_with_probs = [{"car_type": le_car_type.classes_[index], "probability": round(probas[index]*100)} for index in top_indices]

        return Response(top_cars_with_probs)




@api_view(['GET'])
def predict_profit(request):
    try:
        # Connection to the database and data extraction
        connection = mysql.connector.connect(host='localhost', user='root', database='dw_abracadata')
        sql = """
        SELECT 
            r.total_amount, 
            r.price_charged, 
            ca.car_type, 
            c.partner_name, 
            r.profit,
            r.consommation_price,
            r.consommation_moy_carburant_l_par_100km,
            r.emission_co2,
            r.partner_cost,
            r.charges
        FROM 
            factrequest r
        JOIN 
            dimpartners c ON r.partner_fk = c.partner_pk
        JOIN 
            dimcars ca ON r.car_fk = ca.car_pk
        """
        df = pd.read_sql(sql, con=connection)
        connection.close()

        # Preprocessing with OneHotEncoder for categorical variables
        categorical_features = ['partner_name']
        numeric_features = ['total_amount', 'price_charged', 'consommation_price', 'consommation_moy_carburant_l_par_100km', 'emission_co2', 'partner_cost']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Creating the linear regression pipeline
        regressor = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Selecting features (X) and label (y)
        X = df.drop('profit', axis=1)
        y = df['profit']

        # Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the linear regression model
        regressor.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results_df = pd.DataFrame({
            'partner_name': X_test['partner_name'].reset_index(drop=True),
            'Actual Profit': y_test.reset_index(drop=True),
            'Predicted Profit': y_pred
        })

        # Returning the results
        return Response({
            'results': results_df.to_dict(),
            'mse': mse,
            'r2': r2
        })

    except Exception as e:
        return Response({'error': str(e)})


@api_view(['GET'])
def predict_cartype_profit(request):
    try:
        # Connection to the database and data extraction
        connection = mysql.connector.connect(host='localhost', user='root', database='dw_abracadata')
        sql = """
        SELECT 
            r.total_amount, 
            r.price_charged, 
            ca.car_type, 
            ca.matricule,
            c.client_name, 
            r.profit,
            r.consommation_price,
            r.consommation_moy_carburant_l_par_100km,
            r.emission_co2,
            r.charges
        FROM 
            factrequest r
        JOIN 
            dimclients c ON r.client_fk = c.client_pk
        JOIN 
            dimcars ca ON r.car_fk = ca.car_pk
        """
        df = pd.read_sql(sql, con=connection)
        connection.close()

        # Preprocessing with OneHotEncoder for categorical variables
        categorical_features = ['car_type']
        numeric_features = ['total_amount', 'price_charged', 'consommation_price', 'consommation_moy_carburant_l_par_100km', 'emission_co2']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Creating the linear regression pipeline
        regressor = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Selecting features (X) and label (y)
        X = df.drop('profit', axis=1)
        y = df['profit']

        # Division of data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the linear regression model
        regressor.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results_df = pd.DataFrame({
            'Car Type': X_test['car_type'].reset_index(drop=True),
            'matricule':X_test['matricule'].reset_index(drop=True),
            'Actual Profit': y_test.reset_index(drop=True),
            'Predicted Profit': y_pred
        })

        # Returning the results
        return Response({
            'results': results_df.to_dict(),
            'mse': mse,
            'r2': r2
        })

    except Exception as e:
        return Response({'error': str(e)})
    


@api_view(['GET'])
def predict_client_profit(request):
    try:
        # Connection to the database and data extraction
        connection = mysql.connector.connect(host='localhost', user='root', database='dw_abracadata')
        sql = """
        SELECT
            r.total_amount, 
            r.price_charged, 
            ca.car_type, 
            c.client_name, 
            r.profit,
            r.consommation_price,
            r.consommation_moy_carburant_l_par_100km,
            r.emission_co2,
            r.charges
        FROM 
            factrequest r
        JOIN 
            dimclients c ON r.client_fk = c.client_pk
        JOIN 
            dimcars ca ON r.car_fk = ca.car_pk
        """
        df = pd.read_sql(sql, con=connection)
        connection.close()

        # Preprocessing with OneHotEncoder for categorical variables
        categorical_features = ['client_name']
        numeric_features = ['total_amount', 'price_charged', 'consommation_price', 'consommation_moy_carburant_l_par_100km', 'emission_co2']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Creating the linear regression pipeline
        regressor = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Selecting features (X) and label (y)
        X = df.drop('profit', axis=1)
        y = df['profit']

        # Division of data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the linear regression model
        regressor.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results_df = pd.DataFrame({
            'client name': X_test['client_name'].reset_index(drop=True),
            'Actual Profit': y_test.reset_index(drop=True),
            'Predicted Profit': y_pred
        })

        # Returning the results
        return Response({
            'results': results_df.to_dict(),
            'mse': mse,
            'r2': r2
        })

    except Exception as e:
        return Response({'error': str(e)})

# views.py


@api_view(['POST'])
def predict_request_types(request):
    try:
        # Get the future date from the request data
        future_date = request.data.get('future_date')

        # Setup database connection
        engine = create_engine('mysql+mysqlconnector://root@localhost/dw_abracadata')

        # Define the SQL query
        query = """
        SELECT 
            da.date AS ds,
            rt.req_type,
            COUNT(*) AS y
        FROM 
            factrequest fr
        JOIN dimdates da ON fr.arrivals_date_fk = da.date
        JOIN dimrequesttypes rt ON fr.req_fk = rt.req_type_pk
        GROUP BY 
            da.date, rt.req_type;
        """

        # Read the data into a DataFrame
        data = pd.read_sql_query(query, engine)

        # Pivot the data to create a separate time series for each request type
        pivot_data = data.pivot_table(index='ds', columns='req_type', values='y', aggfunc='sum').fillna(0)

        # Reset index to use 'ds' as a column
        pivot_data.reset_index(inplace=True)

        # Initialize a dictionary to store models and forecasts
        models = {}
        forecasts = {}

        # Split data and train a model for each request type
        for req_type in pivot_data.columns[1:]:  # Exclude 'ds' column
            # Prepare the dataset for the request type
            temp_df = pivot_data[['ds', req_type]].rename(columns={req_type: 'y'})
            
            # Initialize the Prophet model with custom parameters
            model = Prophet(
                growth='linear',  # or 'logistic'
                yearly_seasonality=True,  # or False
                weekly_seasonality=True,  # or False
                holidays_prior_scale=10.0  # Adjust the strength of the holiday effect
            )
            
            # Fit the model on the dataset
            model.fit(temp_df)
            
            # Store the model
            models[req_type] = model
            
            # Extend the dataframe to cover the future date
            future_dates = pd.DataFrame({'ds': [future_date]})
            
            # Make predictions for the future date
            forecast = model.predict(future_dates)
            
            # Calculate the range based on the confidence intervals
            upper_bound = round(max(0, forecast.loc[0, 'yhat_upper']))
            
            # Store prediction range
            forecasts[req_type] = (upper_bound)

        # Close the database connection
        engine.dispose()

        # Return the predictions
        return Response(forecasts)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    


from django.core.files.base import ContentFile

@api_view(['POST'])
def detect_car(request):
    if request.method == 'POST' and 'image' in request.FILES:
        try:
            # Get the image file from request.FILES
            image_file = request.FILES['image']

            # Read image content
            image_content = image_file.read()

            # Open and process the image
            img = Image.open(ContentFile(image_content))

            # Configure GenerativeAI
            os.environ["API_KEY"] = "AIzaSyCeFiTQ-y7TP7HfVKWvyVo4jRxn5CoRIJA"  
            genai.configure(api_key=os.environ["API_KEY"])
            model = genai.GenerativeModel('gemini-pro-vision')

            # Define text for model
            text = "Whats the brand , the model and the plate number of the car ? (response only the brand in first line , the model in the second line and the plate number in the third line of the car)"

            # Generate content
            response = model.generate_content([text, img], stream=True)
            response.resolve()
            print(response.text)

            # Split response text to extract brand, model, and plate number
            brand, model, plate_number = response.text.strip().split('\n')

            # Create a Dimcars object with the extracted details
            Dimcars.objects.create(car_type=brand.strip(), modele=model.strip(), matricule=plate_number.strip() ,car_owner = 'SPN')

            return Response({
                "car_type": brand.strip(),
                "modele": model.strip(),
                "matricule": plate_number.strip()
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'No image file found in request.'}, status=400)
    

@csrf_exempt
def forecast_plot(request):
    if request.method == 'POST':
        # Récupérer les données de la requête POST
        post_data = json.loads(request.body.decode('utf-8'))
        num_forecast_months = post_data.get('num_forecast_steps')
        
        # Importer les données de la base de données
        engine = create_engine('mysql://root:@localhost/dw_abracadata')
        query = """
        SELECT f.arrivals_date_fk AS date, SUM(f.total_amount) as daily_total
        FROM factrequest f
        JOIN dimclients c ON f.client_fk = c.client_pk
        JOIN dimdates d on c.arrivals_date = d.date
        GROUP BY c.arrivals_date
        ORDER BY c.arrivals_date;
        """
        df = pd.read_sql(query, engine)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        df_aggregated = df.resample('D').sum()
        
        auto_model = auto_arima(df_aggregated['daily_total'], seasonal=True, m=7, trace=True,
                                error_action='ignore', suppress_warnings=True)
        
        # Forecast future values
        forecast = auto_model.predict(n_periods=num_forecast_months*30)  # Assuming 30 days per month
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=num_forecast_months*30, freq='D')
        
        # Combine future dates with forecasted values
        forecast_df = pd.DataFrame({
            'Predicted_Total': forecast
        }, index=future_dates)
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(df_aggregated.index, df_aggregated['daily_total'], label='Historical Daily Total')
        plt.plot(forecast_df.index, forecast_df['Predicted_Total'], label='Forecasted Daily Total', color='red')
        plt.legend()
        plt.title('Forecast vs Actuals')
        
        plot_path = r'C:\Users\khmir\Desktop\pi\Frontend-pibi\src\assets\forecast_plot.png'

        plt.savefig(plot_path)
        plt.close()
        
        return HttpResponse(status=200)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=400)
