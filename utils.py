def print_progress_bar(i, n, w=50):
    import sys
    if i % (n / 100) == 0:
        p = float(i) / n
        m = int(p * w)
        print("{:>5.0%} [{}]\r".format(p, "=" * m + " " * (w - m))),
        sys.stdout.flush()

def stop_progress_bar(w=50):
    import sys
    print("{:>5.0%} [{}]".format(1.0, "=" * w))
    sys.stdout.flush()

def yaml_annotation_reader(f, depth, with_progress_bar=False):
    import yaml
    try:
        from yaml import cLoader as Loader
    except ImportError:
        from yaml import Loader

    if with_progress_bar:
        update_progress_bar = print_progress_bar
        finalize_progress_bar = stop_progress_bar
    else:
        update_progress_bar =lambda i, n, w=50: None
        finalize_progress_bar = lambda w=50: None

    s_item = ""
    flist = list(f)
    n = len(flist)
    for i, s in enumerate(flist):
        update_progress_bar(i, n)
        if s.startswith(" " * depth + "-"):
            if s_item != "":
                x = yaml.load(s_item, Loader=Loader)
                yield {"image": x[0]["sources"][0]["path"], "label": x[0]["bounding_boxes"][0]["label"]}
                del x
            s_item = ""
        s_item += s
    if s_item != "":
        x = yaml.load(s_item, Loader=Loader)
        yield {"image": x[0]["sources"][0]["path"], "label": x[0]["bounding_boxes"][0]["label"]}
        del x
    finalize_progress_bar()

def load_annotation(file_path):
    annotation = []
    with file(file_path, "r") as f:
        f.readline()
        f.readline()
        for a in yaml_annotation_reader(f, 3):
            annotation.append(a)
    return annotation