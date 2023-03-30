def put_to_same_device(*args):
    device = None
    for arg in args:
        if arg is not None:
            device = arg.device
            break
    if device is None:
        return args
    else:
        return tuple(arg.to(device) for arg in args)

def uniform_dtype(args, dtype='torch.float16'):
    return tuple(arg.to(dtype) for arg in args)