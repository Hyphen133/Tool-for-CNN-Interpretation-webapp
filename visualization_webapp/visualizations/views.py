import subprocess
import torch
from PIL import Image
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from plugins_management import get_plugin_repository, PluginSelector
from torch_model_loading import ModelLoader
from torchvision import transforms
from visualization_core import GraphExtractor, FunctionNode, GraphVisualizationAttacher, GraphUtils
from visualization_printing import LinkAttacher, NodeColoringTool, GraphPrinter
from visualization_printing.maps_printer import save_tensor_as_image
from visualization_utils.image_processing import ImageProcessing

file_lines = []
parent_node = None
selected_plugin_names = None


def index(request):
    return render(request, 'start.html', context={})


def selection(request):
    global parent_node
    global selected_plugin_names

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
            # TODO modify for nongrapgh ones
            GraphVisualizationAttacher.attach_visualizations_to_graph(parent_node, plugin.name,
                                                                      plugin.get_module_visualizations_list_map(model,
                                                                                                                image_tensor))

        return redirect('/static/output_graph.svg')

    plugin_names = PluginSelector.get_all_plugin_names()

    return render(request, 'selection_page.html', context={'plugin_names': plugin_names})


def visualization_page(request):
    print('Inside Visualization page')
    return render(request, 'graph_visualization_page.html', context={})


def node_visualization_page(request, id=0, plugin_name=''):
    global selected_plugin_names
    global parent_node

    if plugin_name == '':
        plugin_name = selected_plugin_names[0]

    node = GraphUtils.find_node_by_id(parent_node, id)

    # Save all visulizations with naming conventions: {node_id}_{plugin_name}_{map_index} in /static/visualizations
    for visualizations_maps in node.get_visualization_maps():
        for i, map in enumerate(visualizations_maps.get_map_list()):
            img = transforms.ToPILImage()(map).convert('LA')
            img.save('./static/visualizations/' + str(node.id) + "_" + visualizations_maps.group_name + "_" + str(
                i + 1) + '.png')

    map_links = [[(1, "https://granty.pl/wp-content/uploads/2017/03/ohio-114098_1920-600x400.jpg"),
                  (2, "https://multidom.pl/thumbs/fit-600x400/2015-10::1444833334-17.jpg"),
                  (
                      3,
                      "https://esero.kopernik.org.pl/wp-content/uploads/2019/05/slajder_2_galaktyka_kobiet-600x400.jpg"),
                  (4, "http://www.informationclearinghouse.info/Nuclear-Bomb-New-York-Public-Domain-600x400.jpg"),
                  (5, "https://upload.wikimedia.org/wikipedia/commons/6/69/600x400_kastra.jpg"),
                  (6, "https://www.ledhut.co.uk/blog/wp-content/uploads/2015/09/Glasgow-600x400.jpg")
                  ],
                 [(7, "http://www.informationclearinghouse.info/Nuclear-Bomb-New-York-Public-Domain-600x400.jpg"),
                  (8, "https://upload.wikimedia.org/wikipedia/commons/6/69/600x400_kastra.jpg"),
                  (9, "https://www.ledhut.co.uk/blog/wp-content/uploads/2015/09/Glasgow-600x400.jpg"),
                  (10, "http://www.informationclearinghouse.info/Nuclear-Bomb-New-York-Public-Domain-600x400.jpg"),
                  (11, "https://upload.wikimedia.org/wikipedia/commons/6/69/600x400_kastra.jpg"),
                  (12, "https://www.ledhut.co.uk/blog/wp-content/uploads/2015/09/Glasgow-600x400.jpg")
                  ]]

    return render(request, 'node_visualization_page.html',
                  context={'current_plugin_name': plugin_name, 'node_id': node.id,
                           'plugin_names': selected_plugin_names, "links": map_links,
                           "links_count": sum([len(x) for x in map_links]), })


def node_redirect(request, id=0):
    global selected_plugin_names
    return node_visualization_page(request, id, selected_plugin_names[0])
