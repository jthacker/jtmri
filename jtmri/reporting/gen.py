import argparse, os
from jinja2 import Environment, PackageLoader


def generate(path, templates):
    env = Environment
    template = env.get_template(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Report')
    parser.add_argument('path', help='path to template to generate report from',
            type=argparse.FileType(mode='r'))
    parser.add_argument('includes', help='path to a directory of templates that may be included in the base template')
    args = parser.parse_args()
    generate(path)
