import json
from pprint import pprint


def custom_pretty_print(title, data, width=120, is_line=True):
    print(title)
    pprint(data, width=width)
    if is_line:
        print("-"*width)


def pretty_print_json(json_sections):
    res = json.dumps(json_sections, ensure_ascii=False, indent=4)
    print(res)