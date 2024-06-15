from django.urls import path
from . import views
from .views import CarRecommendationAPIView

urlpatterns = [
    path('', views.apiOverview, name='apiOverview'),
    path('user-list/', views.ShowAll, name='user-list'),
    path('create-user/', views.CreateUser, name='user-create'),
    path('view-user/<int:pk>/', views.ViewUser, name='view-user'),
    path('user-update/<int:pk>/', views.updateUser, name='user-update'),
    path('user-delete/<int:pk>/', views.deleteUser, name='user-delete'),
    path('login/', views.UserLogin, name='login'),
    path('scoring/', views.scoring, name='scoring'),
    path('predict-profit/', views.predict_profit, name='predict_profit'),
    path('predict-cartype_profit/', views.predict_cartype_profit, name='cartype_profit'),
    path('predict_client_profit/', views.predict_client_profit, name='predict_client_profit'),
    path('car-recommendations/', CarRecommendationAPIView.as_view(), name='car_recommendations'),
    path('detect-car/', views.detect_car, name='detect_car'),
    path('serie-temp/' ,views.forecast_plot, name='serietemp'),
     path('predict-request-types/', views.predict_request_types, name='predict_request_types'),
     ]


   
