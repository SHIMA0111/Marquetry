import os
import subprocess

from PIL import Image


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    """Plot a computational graph as a DOT graph and save it to an image file.

        This function generates a DOT graph representing the computational graph backward-reachable
        from the given output container.
        The DOT graph is then saved as an image file in the specified format.

        Args:
            output (:class:`marquetry.Container`): The output container of the computational graph to visualize.
            verbose (bool, optional): If True, includes detailed information
                about container names, shapes, and data types in the DOT graph.
                Default is True.
            to_file (str): The file path where the DOT graph image should be saved.
                Default is "graph.png".

        Returns:
            None
    """

    dot_graph = get_dot_graph(output.node, verbose)

    tmp_dir = os.path.join(os.path.expanduser("~"), ".module_tmp")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    img = Image.open(to_file)
    img.show()


def get_dot_graph(output, verbose=True):
    """Generates a graphviz DOT text of a computational graph.

        Build a graph of functions and containers backward-reachable from the output.
        To visualize a graphviz DOT text, you need the dot binary from the graph viz
        package (www.graphviz.org).
    """
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + "}"


def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        ret += dot_edge.format(id(f), id(y()))

    return ret
