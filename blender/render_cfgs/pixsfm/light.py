import json
light_cfg = {"mesh": {"pose": [[20, -150, -50], ], 
                      "energy": [0.9, ], 
                      "type":["SUN", ], 
                      "scale":[1, ], 
                      "rotation":[[1.571, 0.0, 0.0], ]}, 
            "texture": {"pose": [[20, -150, -50], ], 
                        "energy": [5, ],
                        "type":["SUN", ], 
                        "scale":[1, ], 
                        "rotation":[[1.571, 0.0, 0.0], ]}}

with open("light.json", "w") as f:
    json.dump(light_cfg, f)