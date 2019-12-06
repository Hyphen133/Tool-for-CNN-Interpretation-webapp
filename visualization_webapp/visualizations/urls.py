from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('visualization/', views.visualization_page, name='visualization'),
]