from os import path

import tensorflow as tf
from tensorflow.python.framework import graph_util


def export_graph(input_path, output_path, output_nodes, debug=False):
    # todo: might want to look at http://stackoverflow.com/a/39578062/195651

    checkpoint = tf.train.latest_checkpoint(input_path)
    importer = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)

    graph = tf.get_default_graph()  # type: tf.Graph
    gd = graph.as_graph_def()  # type: tf.GraphDef

    if debug:
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

    if debug:
        print('Freezing the graph ...')
    with tf.Session() as sess:
        importer.restore(sess, checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(sess, gd, output_nodes)
        tf.train.write_graph(output_graph_def, path.dirname(output_path), path.basename(output_path), as_text=False)


def import_graph(input_path, output_nodes=None, input_map=None, debug=False, use_current_graph=True, prefix='imported'):
    if debug:
        print('Reading back the graph ...')

    with tf.gfile.GFile(input_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.get_default_graph() if use_current_graph else tf.Graph()
    with graph.as_default():
        tensors = tf.import_graph_def(
                graph_def,
                input_map=input_map,
                return_elements=output_nodes,
                name=prefix,
                op_dict=None,
                producer_op_list=None
            )

    # the output argument corresponds to the tensors defined by return_elements
    assert (tensors is None and output_nodes is None) or (len(tensors) == len(output_nodes))

    if debug:
        ops = graph.get_operations()
        op_names = [op.name for op in ops]
        print(op_names)

    return tensors
