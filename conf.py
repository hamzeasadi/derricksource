import os




root = os.getcwd()
datapath = os.path.join(root, 'data')
paths = dict(
    root=root, data=datapath, model=os.path.join(datapath, 'model'), liebherr=os.path.join(datapath, 'liebherr'), vision=os.path.join(datapath, 'vision'), 
    
    # visiontrainiframes = os.path.join('/Users/hamzeasadi/python/resnetsource/data/vision/visioniframes/visiontrainiframes'), 
    # visiontestiframes = os.path.join('/Users/hamzeasadi/python/resnetsource/data/vision/visioniframes/visiontestiframes'),

    visiontrainiframes = os.path.join('/home/hasadi/project/resnetsource/data/vision/visioniframes/visiontrainiframes'), 
    visiontestiframes = os.path.join('/home/hasadi/project/resnetsource/data/vision/visioniframes/visiontestiframes'),

    visiontrain = os.path.join(datapath, 'vision', 'visiontrain'),
    visiontest = os.path.join(datapath, 'vision', 'visiontest'),

    # liebherrtrainiframes = os.path.join('/Users/hamzeasadi/python/resnetsource/data/liebherr/liebherriframes/liebherrtrainiframes'),
    # liebherrtestiframes = os.path.join('/Users/hamzeasadi/python/resnetsource/data/liebherr/liebherriframes/liebherrtestiframes'),

    liebherrtrainiframes = os.path.join('/home/hasadi/project/resnetsource/data/liebherr/liebherriframes/liebherrtrainiframes'),
    liebherrtestiframes = os.path.join('/home/hasadi/project/resnetsource/data/liebherr/liebherriframes/liebherrtestiframes'),

    liebherrtrain = os.path.join(datapath, 'liebherr', 'liebherrtrain'),
    liebherrtest = os.path.join(datapath, 'liebherr', 'liebherrtest'),
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