from django.shortcuts import render, HttpResponse
from django.views.generic.base import TemplateView


class Home(TemplateView):
    template_name = 'core/home.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)

class fromCamera(TemplateView):
    template_name = 'core/camera.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)

class fromFile(TemplateView):
    template_name = 'core/selectfile.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)

class fromFile_afterclick(TemplateView):
    template_name = 'core/file.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)

class Filter(TemplateView):
    template_name = 'core/filter.html'
    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)
