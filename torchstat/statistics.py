import torch
import torch.nn as nn
from torchstat import ModelHook
from collections import OrderedDict
from torchstat import StatTree, StatNode, report_format

def get_parent_node(root_node, stat_node_name):
    assert isinstance(root_node, StatNode)

    node = root_node
    names = stat_node_name.split('.')
    for i in range(len(names) - 1):
        node_name = '.'.join(names[0:i+1])
        child_index = node.find_child_index(node_name)
        assert child_index != -1
        node = node.children[child_index]
    return node


def convert_leaf_modules_to_stat_tree(leaf_modules):
    assert isinstance(leaf_modules, OrderedDict)

    create_index = 1
    root_node = StatNode(name='root', parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        names = leaf_module_name.split('.')
        for i in range(len(names)):
            create_index += 1
            stat_node_name = '.'.join(names[0:i+1])
            parent_node = get_parent_node(root_node, stat_node_name)
            is_leaf = i == len(names) - 1
            node = StatNode(name=stat_node_name, parent=parent_node, is_leaf=is_leaf)
            parent_node.add_child(node)
            if i == len(names) - 1:  # leaf module itself
                input_shape = leaf_module._input_shape.numpy().tolist()
                output_shape = leaf_module._output_shape.numpy().tolist()
                node.input_shape = input_shape
                node.output_shape = output_shape
                node.parameter_quantity = leaf_module.parameter_quantity.numpy()[0]
                node.inference_memory = leaf_module.inference_memory.numpy()[0]
                node.MAdd = leaf_module.MAdd.numpy()[0]
                node.Flops = leaf_module.Flops.numpy()[0]
                node.duration = leaf_module.duration.numpy()[0]
                node.Memory = leaf_module.Memory.numpy().tolist()
                node.update_leaf_child()
    return StatTree(root_node)


class ModelStat(object):
    def __init__(self, model, example_input, query_granularity=1, model_fn: str = 'forward'):
        assert isinstance(model, nn.Module)
        self._model = model
        self._example_input = example_input
        self._query_granularity = query_granularity
        self.model_fn = model_fn
    def _analyze_model(self):
        model_hook = ModelHook(self._model, self._example_input, model_fn=self.model_fn)
        self.leaf_modules = model_hook.retrieve_leaf_modules()
        self.stat_tree = convert_leaf_modules_to_stat_tree(self.leaf_modules)
        collected_nodes = self.stat_tree.get_collected_stat_nodes(self._query_granularity)
        return collected_nodes

    def show_report(self):
        collected_nodes = self._analyze_model()
        report, report_df = report_format(collected_nodes)
        return report, report_df


def stat(model, example_input, query_granularity=1, model_fn: str = 'forward', simple: bool = False):
    ms = ModelStat(model, example_input, query_granularity, model_fn)
    report = ms.show_report()[0]
    if simple:
        # parse the flops, param, memory from the report
        parsed_report = report.split('\n')
        params_line = parsed_report[-7].replace('Total', '')
        memory_line = parsed_report[-5].replace('Total', '')
        madd_line = parsed_report[-4].replace('Total', '')
        flops_line = parsed_report[-3].replace('Total', '')
        print(' | '.join([params_line, memory_line,madd_line, flops_line]))
    else:
        print(report)
