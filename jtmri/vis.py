from contextlib import nested
from tempfile import NamedTemporaryFile as TmpFile
from subprocess import check_call
from IPython.display import Image


def display_dot(dot_data):
    '''Display a graphviz dot data
    Args:
        dot_data -- file like object that contains dot data

    Retruns: IPython display Image for viewing in a notebook
    '''
    with nested(TmpFile(), TmpFile()) as (input_file, output_file):
        input_file.write(dot_data.read())
        input_file.flush()
        cmd = ['dot', '-Tpng', input_file.name, '-o', output_file.name]
        check_call(cmd)
        return Image(output_file.read())


def display_sklearn_tree(tree):
    '''Display a sklearn tree as a graph'''
    from sklearn.tree import export_graphviz
    from StringIO import StringIO
    input_file = StringIO()
    export_graphviz(tree, input_file)
    input_file.seek(0)
    return display_dot(input_file)
