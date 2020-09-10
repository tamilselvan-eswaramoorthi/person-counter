from django.conf.urls import url
from . import views
urlpatterns = [
    url(r'^$', views.index),
    url(r'^fail$', views.fail),
    url(r'^login$', views.login),
    url(r'^filter$', views.filter),
    url(r'^selectfile_fn$', views.selectfile)

	]