import json
light_cfg = {'mesh':{'pose':((1, 0, -2), (0, 1, -2), (0, -1, -2)), 'energy':(12.0, 8.0, 7.0)}, 'texture':{'pose':((1, 0, -2), (0, 1, -2), (0, -1, -2)), 'energy':(70.0, 45.0, 45.0)}}

with open("light.json", "w") as f:
    json.dump(light_cfg, f)