from libs.impex import import_graph, export_graph


if __name__ == '__main__':
    input_path = 'log/20170205-034325-2'
    output_path = 'exported/vae.pb'

    # define the nodes required for inference
    output_nodes = ['vae/variational/Reshape', 'vae/decoder/6/Elu']
    export_graph(input_path, output_path, output_nodes, debug=True)

    # define the nodes required for usage
    required_nodes = ['vae/x'] + output_nodes
    import_graph(output_path, required_nodes, debug=True, use_current_graph=False)
