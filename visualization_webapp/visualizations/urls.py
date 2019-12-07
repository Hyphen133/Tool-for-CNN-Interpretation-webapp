from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('selection/', views.selection, name='selection'),
    path('visualization/', views.visualization_page, name='visualization'),
    path('node/<id>/', views.node_redirect, name='visualization'),
    path('node/<id>/<plugin_name>', views.node_visualization_page, name='node'),
]