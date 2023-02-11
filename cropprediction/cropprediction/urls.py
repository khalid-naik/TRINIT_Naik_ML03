# from django.contrib import admin

from django.urls import path, include
from cpapp import views

urlpatterns = [
    path('', views.input_form, name='input_form'),
    path('predict/', views.predict, name='predict'),
]


