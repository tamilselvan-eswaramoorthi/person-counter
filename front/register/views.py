from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages

from .models import User
import pandas as pd
from datetime import datetime

def index(request):
    return render(request, 'register/index.html')


def login(request):
    mail, password =  (request.POST['login_email']), (request.POST['login_password'])
    if mail == 'admin@email.com' and password == '1234':
        return redirect('/home')
    return redirect('/fail')

def fail(request):
    return render(request, 'register/index.html')

def filter(request):
    date, stime, etime =  ((request.POST['date']), (request.POST['start time']), (request.POST['end time']))
    date, stime, etime = str(date), str(stime), str(etime)
    stime  = stime + ":00" ; etime  = etime + ":00" ;
    date = datetime.strptime(date, '%Y-%m-%d')
    stime = datetime.strptime(stime, '%H:%M:%S')
    etime = datetime.strptime(etime, '%H:%M:%S')
    df = pd.read_csv('resources/log.csv')
    df['date'] = pd.to_datetime(df['date'], format= '%Y-%m-%d').apply(pd.Timestamp)
    df['time'] = pd.to_datetime(df['time'], format= '%H:%M:%S').apply(pd.Timestamp)
    date_df = df[df['date'] == date]
    time_df = date_df[date_df['time'] > stime]
    time_df = time_df[time_df['time'] < etime]
    time_df['time'] = time_df['time'].apply(lambda x: x.strftime('%H:%M:%S'))
    return HttpResponse(time_df.to_html())

def selectfile(request):
    return redirect('/file')
