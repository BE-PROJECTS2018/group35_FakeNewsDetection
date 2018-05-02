from django.urls import path,include
from app import views
urlpatterns = [
	path(r'', views.index, name='index'),

]