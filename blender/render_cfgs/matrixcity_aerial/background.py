import json

back_sphere_radius = 7.5

with open("background.json", "w") as f:
    json.dump(back_sphere_radius, f)