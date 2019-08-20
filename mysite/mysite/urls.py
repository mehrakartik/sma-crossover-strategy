"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import stock

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', stock.onHome, name='home'),
    path('submit/', stock.onSubmit, name='submit'),
    path('remove/', stock.onRemoveComparison, name='remove'),
    path('compare/', stock.onCompare, name='compare'),
    path('smacs/', stock.onSMA, name='smacs'),
    path('home/', stock.onHome, name='home'),
    path('backtest/', stock.onBacktest, name='backtest'),
    # path('aboutus/',views.fun),
    path('trending/',stock.onTrending,name='onTrending'),
    # path('getData/',views.getData,name='getData'),
    # path('getimage/',views.getimage,name='getimage'),

]