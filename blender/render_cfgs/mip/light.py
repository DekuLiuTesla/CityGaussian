import json
light_cfg = {'mesh':{'pose':((2., -3.5, 0), (-1, -3, 1), (0, -3, -1)), 'energy':(60.0, 50.0, 30.0)}, 'texture':{'pose':((2., -3.5, 0), (-1, -3, 1), (0, -3, -1)), 'energy':(200.0, 150.0, 150.0)}}

with open("light.json", "w") as f:
    json.dump(light_cfg, f)