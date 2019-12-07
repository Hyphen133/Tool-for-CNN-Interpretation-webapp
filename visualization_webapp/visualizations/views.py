import subprocess
import torch
from PIL import Image
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from plugins_management import get_plugin_repository, PluginSelector
from torch_model_loading import ModelLoader
from visualization_core import GraphExtractor, FunctionNode, GraphVisualizationAttacher, GraphUtils
from visualization_printing import LinkAttacher, NodeColoringTool, GraphPrinter
from visualization_utils.image_processing import ImageProcessing

file_lines = []
parent_node = None

def index(request):
    plugin_names = ['feature_maps', 'gradient_maps', 'filters']
    return render(request, 'hello.html', context={'plugin_names': plugin_names})

def selection(request):
    global parent_node

    if request.method == 'POST':
        print("Inside selection view")

        ############     Loading files and saving files
        uploaded_file = request.FILES['code']
        uploaded_image = request.FILES['image']
        img = Image.open(uploaded_image)
        img.save('./static/image.jpg')

        for l in uploaded_file.readlines():
            file_lines.append(l)


        file_name = default_storage.save('model.py', uploaded_file)

        #################   Loading model from file
        model_loader = ModelLoader()
        model = model_loader.load_model_from_external_file(file_name)
        input_shape = model_loader.load_input_shape_from_external_file(file_name)
        print(model)

        #################   Converting model to graph
        test_input = torch.rand(1, input_shape[0], input_shape[1], input_shape[2])
        y = model.forward(test_input)
        graph_extractor = GraphExtractor()
        parent_node = FunctionNode(y.grad_fn.__class__.__name__)
        graph_extractor.create_graph_and_associate_with_mapping(parent_node, y.grad_fn.next_functions, model, y.grad_fn)

        link_attacher = LinkAttacher("/node")
        coloring_tool = NodeColoringTool("blue", "lightyellow")
        graph_printer = GraphPrinter(link_attacher, coloring_tool)
        dot_text = graph_printer.convert_graph_to_dot(parent_node)

        with open('graph.dot', 'w') as f:
            f.writelines(dot_text)

        subprocess.check_call("dot -Tsvg graph.dot -o ./static/output_graph.svg")

        #############   Attach visualization by selected nodes
        image_tensor = ImageProcessing.pil_img_to_tensor_of_with_size(img, input_shape).unsqueeze(0)

        selected_plugin_names = request.POST.getlist('selected_plugins')
        selected_plugins = PluginSelector.get_selected_plugins(selected_plugin_names)

        for plugin in selected_plugins:
            #TODO modify for nongrapgh ones
            GraphVisualizationAttacher.attach_visualizations_to_graph(parent_node, plugin.name, plugin.get_module_visualizations_list_map(model, image_tensor))

        return redirect('/static/output_graph.svg')



    plugin_names = PluginSelector.get_all_plugin_names()

    return render(request, 'selection_page.html', context={'plugin_names': plugin_names})


def visualization_page(request):
    print('Inside Visualization page')
    return render(request, 'graph_visualization_page.html', context={})


def node_visualization_page(request,id=0):
    #flatten and load by id
    global parent_node

    node = GraphUtils.find_node_by_id(parent_node, id)



    return render(request, 'node_visualization_page.html', context={'id' : id})
