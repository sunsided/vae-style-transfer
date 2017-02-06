from os import path

import tensorflow as tf
from tensorflow.python.framework import graph_util


def export_graph(input_path, output_path, output_nodes):
    # import the VAE metagraph
    checkpoint = tf.train.latest_checkpoint(input_path)
    importer = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)

    graph = tf.get_default_graph()  # type: tf.Graph
    gd = graph.as_graph_def()  # type: tf.GraphDef

    op_names = [op.name for op in graph.get_operations()]
    print(op_names)

    # fix batch norm nodes
    # https://github.com/tensorflow/tensorflow/issues/3628
    for node in gd.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] += '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    print('Freezing the graph ...')
    with tf.Session() as sess:
        importer.restore(sess, checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                     gd,
                                                                     output_nodes)

        tf.train.write_graph(output_graph_def, path.dirname(output_path), path.basename(output_path), as_text=False)


def import_graph(output_path, output_nodes):
    print('\nReading back the graph ...')
    tf.reset_default_graph()

    with tf.gfile.GFile(output_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        prefix = 'imported'
        tensors = tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=output_nodes,
                name=prefix,
                op_dict=None,
                producer_op_list=None
            )

        # the output argument corresponds to the tensors defined
        # by return_elements
        assert len(tensors) == len(output_nodes)

    ops = graph.get_operations()

    op_names = [op.name for op in ops]
    print(op_names)


if __name__ == '__main__':
    input_path = 'log/20170205-034325-2'
    output_path = 'exported/vae.pb'

    # define the nodes required for inference
    output_nodes = ['vae/variational/Reshape', 'vae/decoder/6/Elu']
    export_graph(input_path, output_path, output_nodes)

    # define the nodes required for usage
    required_nodes = ['vae/x'] + output_nodes
    import_graph(output_path, required_nodes)
