import laspy
import warnings
import numpy as np

def read_las_fit(filename, attrs=None):
    """
    Args:
        filename: <str> las file path
        attrs: <list> additional attributes to read from the las file

    Returns:
        xyz, rgb, attr_dict
    """
    if attrs is None:
        attrs = []

    attrs = list(set(attrs + ["scales", "offsets"]))

    inFile = laspy.read(filename)
    N_points = len(inFile)
    x = np.reshape(inFile.x, (N_points, 1))
    y = np.reshape(inFile.y, (N_points, 1))
    z = np.reshape(inFile.z, (N_points, 1))
    xyz = np.hstack((x, y, z))

    rgb = np.zeros((N_points, 3), dtype=np.uint16)
    if hasattr(inFile, "red") and hasattr(inFile, "green") and hasattr(inFile, "blue"):
        r = np.reshape(inFile.red, (N_points, 1))
        g = np.reshape(inFile.green, (N_points, 1))
        b = np.reshape(inFile.blue, (N_points, 1))
        # i = np.reshape(inFile.Reflectance, (N_points, 1))
        rgb = np.float32(np.hstack((r, g, b))) / 65535
    else:
        print(f"{filename.split('/')[-1]} has no RGB information!")

    attr_dict = {}
    for attr in attrs:
        value = None
        if hasattr(inFile, attr):
            value = getattr(inFile, attr)
        elif hasattr(inFile.header, attr):
            value = getattr(inFile.header, attr)
        else:
            warnings.warn(f"{filename.split('/')[-1]} has no information for {attr}!")

        if hasattr(value, "array"):
            attr_dict[attr] = np.array(value)
        else:
            attr_dict[attr] = value

    return xyz, rgb, attr_dict