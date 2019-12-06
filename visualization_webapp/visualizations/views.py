import subprocess

import torch
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from torch_model_loading import ModelLoader
from visualization_core import GraphExtractor, FunctionNode
from visualization_printing import LinkAttacher, NodeColoringTool, GraphPrinter

selected_plugins = []
file_lines = []

def index(request):
    if request.method == 'POST':
        print("Inside selection view")
        uploaded_file = request.FILES['code']

        for l in uploaded_file.readlines():
            file_lines.append(l)

        plugins = request.POST.getlist('selected_plugins')
        for p in plugins:
            selected_plugins.append(p)

        file_name = default_storage.save('model.py', uploaded_file)

        #################   Loading model from file
        model_loader = ModelLoader()
        model = model_loader.load_model_from_external_file(file_name)
        print(model)

        #################   Converting model to graph
        test_input = torch.rand(1, 3, 32, 32)
        y = model.forward(test_input)
        graph_extractor = GraphExtractor()
        parent_node = FunctionNode(y.grad_fn.__class__.__name__)
        graph_extractor.create_graph_and_associate_with_mapping(parent_node, y.grad_fn.next_functions, model, y.grad_fn)

        link_attacher = LinkAttacher("www.google.com")
        coloring_tool = NodeColoringTool("blue", "lightyellow")
        graph_printer = GraphPrinter(link_attacher, coloring_tool)
        dot_text = graph_printer.convert_graph_to_dot(parent_node)

        with open('graph.dot', 'w') as f:
            f.writelines(dot_text)

        subprocess.check_call("dot -Tsvg graph.dot -o ./static/output_graph.svg")


        # print(selected_plugins)
        # print(request.get_full_path)
        # return redirect('/visualizations/visualization')
        return redirect('/static/output_graph.svg')

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
