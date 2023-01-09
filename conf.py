import os




root = os.getcwd()
datapath = os.path.join(root, 'data')
paths = dict(
    root=root, data=datapath, model=os.path.join(datapath, 'model'), liebherr=os.path.join(datapath, 'liebherr'), vision=os.path.join(datapath, 'vision'), 
    
    visiontrain = os.path.join(datapath, 'vision', 'visiontrain'),
    visiontest = os.path.join(datapath, 'vision', 'visiontest'),

    liebherrtrain = os.path.join(datapath, 'liebherr', 'liebherrtrain'),
    liebherrtest = os.path.join(datapath, 'liebherr', 'liebherrtest'),

    visiontrainiframes = os.path.join(os.pardir, 'resnetsource', 'data', 'vision', 'visioniframes', 'visiontrainiframes'),
    visiontestiframes = os.path.join(os.pardir, 'resnetsource', 'data', 'vision', 'visioniframes', 'visiontestiframes'),
    
    
    liebherrtrainiframes = os.path.join(os.pardir, 'resnetsource', 'data', 'liebherr', 'liebherriframes', 'liebherrtrainiframes'),
    liebherrtestiframes = os.path.join(os.pardir, 'resnetsource', 'data', 'liebherr', 'liebherriframes', 'liebherrtestiframes'),
)


def createdir(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(e)


def all_paths(paths: dict):
    for key, val in paths.items():
        createdir(val)


def main():
    print(root)
    all_paths(paths)


if __name__ == '__main__':
    main()