import json

back_sphere_radius = 3.8

with open("background.json", "w") as f:
    json.dump(back_sphere_radius, f)