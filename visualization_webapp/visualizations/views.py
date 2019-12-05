import requests
from django.core.files import File
from django.shortcuts import render, redirect

from django.core.files.storage import default_storage

selected_plugins = []
file_lines = []

def index(request):
    if request.method == 'POST':
        print("Inside selection view")
        uploaded_file = request.FILES['code']
        # print(uploaded_file.name)
        # print(uploaded_file.size)
        # print()

        for l in uploaded_file.readlines():
            file_lines.append(l)

        plugins = request.POST.getlist('selected_plugins')
        for p in plugins:
            selected_plugins.append(p)

        # file_name = default_storage.save('model.py', uploaded_file)

        print(selected_plugins)
        print(request.get_full_path)
        return redirect('/visualizations/visualization')

    plugin_names = ['feature_maps', 'gradient_maps', 'filters']

    return render(request, 'selection_page.html', context={'plugin_names': plugin_names})


def visualization_page(request):
    print('Inside Visualization page')
    # file = default_storage.open('model.py')
    # print(len(file.readlines()))
    # print(file.readlines())

    print(file_lines)
    print(selected_plugins)

    return render(request, 'graph_visualization_page.html', context={})
