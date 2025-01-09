

def pipeline(data, transforms):
    for transform in transforms:
        if type(transform) is tuple and len(transform) >= 2 and isinstance(transform[1], dict):
            data = transform[0](data, **transform[1])
        else:
            data = transform(data)
    return data