from django.urls import path
from .views import predict_income, home

urlpatterns = [
    path('', home, name='home'),
    path('predict', predict_income, name='predict'),
]
