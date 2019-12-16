import base64
import io
import os
import subprocess

import math
import torch
from PIL import Image
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from plugins_management import get_plugin_repository, PluginSelector
from torch_model_loading import ModelLoader
from torchvision import transforms
from visualization_core import GraphExtractor, FunctionNode, GraphVisualizationAttacher, GraphUtils
from visualization_core.interfaces.VisualizationTechnique import PrintingMode, GraphVisualizationTechnique, \
    NonGraphVisualizationTechnique
from visualization_core.model.graph_model import NonGraphVisualizationMapsContainer
from visualization_printing import LinkAttacher, NodeColoringTool, GraphPrinter
from visualization_printing.maps_printer import save_tensor_as_image, save_tensor_with_heatmap
from visualization_utils.image_processing import ImageProcessing

file_lines = []
parent_node = None
selected_plugin_names = None
class_index = None
non_graph_visualization_maps_container = NonGraphVisualizationMapsContainer()


def index(request):
    return render(request, 'start.html', context={})


def selection(request):
    global parent_node
    global selected_plugin_names
    global class_index
    global non_graph_visualization_maps_container

    if request.method == 'POST':
        print("Inside selection view")
        class_index = int(request.POST.get("class_index", ""))
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
        # GraphUtils.deep_freezing(model)
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
            if issubclass(plugin.__class__, GraphVisualizationTechnique):
                GraphVisualizationAttacher.attach_visualizations_to_graph(parent_node, plugin.name,
                                                                          plugin.get_module_visualizations_list_map(
                                                                              model,
                                                                              image_tensor,
                                                                              convert_class_index_to_one_hot_vector(
                                                                                  y,
                                                                                  class_index)))
            elif issubclass(plugin.__class__, NonGraphVisualizationTechnique):
                non_graph_visualization_maps_container.set_visualizations_maps(plugin.name,
                                                                               plugin.get_additional_visualizations_maps(
                                                                                   model, image_tensor,
                                                                                   convert_class_index_to_one_hot_vector(
                                                                                       y,
                                                                                       class_index)))

        return redirect('/graph')

    plugin_names = PluginSelector.get_all_plugin_names()

    return render(request, 'selection_page.html', context={'plugin_names': plugin_names})


def graph_visualization_page(request):
    global selected_plugin_names
    non_graph_plugin_names = PluginSelector.get_only_selected_nongraph_plugins_names(selected_plugin_names)
    return render(request, 'graph_visualization_page.html', context={"non_graph_plugin_names": non_graph_plugin_names})


def convert_class_index_to_one_hot_vector(model_output, class_index):
    one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
    one_hot_output[0][class_index] = 1
    return one_hot_output


def visualization_page(request):
    print('Inside Visualization page')
    return render(request, 'graph_visualization_page.html', context={})


def node_visualization_page(request, id=0, plugin_name=''):
    global selected_plugin_names
    global parent_node
    input_image = Image.open(open('./static/image.jpg', 'rb'))

    if plugin_name == '':
        plugin_name = PluginSelector.get_only_selected_graph_plugins_names(selected_plugin_names)[0]

    node = GraphUtils.find_node_by_id(parent_node, id)

    # Clear all past visualizations
    for f in os.listdir('./static/visualizations/'):
        os.remove(os.path.join('./static/visualizations/', f))

    # Save all visulizations for given plugin with naming conventions: {node_id}_{plugin_name}_{map_index} in /static/visualizations
    plugin = PluginSelector.get_selected_plugins([plugin_name])[0]
    visualizations_maps = [maps for maps in node.get_visualization_maps() if maps.group_name == plugin_name][0]

    images_uris = []
    if issubclass(plugin.__class__, GraphVisualizationTechnique):
        for i, map in enumerate(visualizations_maps.get_map_list()):

            mode = plugin.get_printing_mode()

            # Changes based on mode
            if mode == PrintingMode.HEAPMAP:
                img = save_tensor_with_heatmap(input_image, map.detach().numpy(), cmap='jet')
            elif mode == PrintingMode.NORMAL:
                img = transforms.ToPILImage()(map).convert('LA')

            # Saving to in-memory uri
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            data_uri = base64.b64encode(buffer.read()).decode('ascii')
            images_uris.append(data_uri)

    # Divide into map_links with format [nx[6x(id,link)]]
    map_links = []
    for i in range(math.ceil(len(images_uris) / 6.0)):
        map_links.append([])
        for j in range(6):
            index = 6 * i + j
            if index < len(images_uris):
                map_links[i].append((index + 1, images_uris[index]))

    return render(request, 'node_visualization_page.html',
                  context={'current_plugin_name': plugin_name, 'node_id': node.id,
                           'plugin_names': PluginSelector.get_only_selected_graph_plugins_names(selected_plugin_names),
                           "links": map_links,
                           "links_count": sum([len(x) for x in map_links]), })


def nongraph_visualization_page(request, plugin_name=''):
    global selected_plugin_names
    global parent_node
    global non_graph_visualization_maps_container

    input_image = Image.open(open('./static/image.jpg', 'rb'))

    if plugin_name == '':
        plugin_name = PluginSelector.get_only_selected_nongraph_plugins_names(selected_plugin_names)[0]

    # Save all visulizations for given plugin with naming conventions: {plugin_name}_{map_index} in /static/visualizations
    plugin = PluginSelector.get_selected_plugins([plugin_name])[0]

    images_uris = []
    if issubclass(plugin.__class__, NonGraphVisualizationTechnique):
        for i, map in enumerate(non_graph_visualization_maps_container.get_visualization_maps(plugin_name)):
            mode = plugin.get_printing_mode()

            # Changes based on mode
            if mode == PrintingMode.HEAPMAP:
                img = save_tensor_with_heatmap(input_image, map.detach().numpy(), cmap='jet')
            elif mode == PrintingMode.NORMAL:
                img = transforms.ToPILImage()(map).convert('LA')

            # Saving to in-memory uri
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            data_uri = base64.b64encode(buffer.read()).decode('ascii')
            images_uris.append(data_uri)

    # Divide into map_links with format [nx[6x(id,link)]]
    map_links = []
    for i in range(math.ceil(len(images_uris) / 6.0)):
        map_links.append([])
        for j in range(6):
            index = 6 * i + j
            if index < len(images_uris):
                map_links[i].append((index + 1, images_uris[index]))

    return render(request, 'nongraph_visualization_page.html',
                  context={'current_plugin_name': plugin_name,
                           'plugin_names': PluginSelector.get_only_selected_nongraph_plugins_names(
                               selected_plugin_names),
                           "links": map_links,
                           "links_count": sum([len(x) for x in map_links]),
                           })


def node_redirect(request, id=0):
    global selected_plugin_names
    return node_visualization_page(request, id,
                                   PluginSelector.get_only_selected_graph_plugins_names(selected_plugin_names)[0])
