import bpy

color_mesh = (1, 0.878, 0.949)
color_texture = (0.5, 0.5, 0.5)

def newMaterial(id):

    mat = bpy.data.materials.get(id)
    if mat is None:
        mat = bpy.data.materials.new(name=id)

    mat.use_nodes = True
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

    return mat


def newShader(id, type, r, g, b):

    mat = newMaterial(id)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')

    if type == "diffuse":
        shader = nodes.new(type='ShaderNodeBsdfDiffuse')
        nodes["Diffuse BSDF"].inputs[0].default_value = (r, g, b, 1)
    else:
        assert False
    links.new(shader.outputs[0], output.inputs[0])

    return mat


def draw_background(is_texture):
    
    if is_texture:
        mat = newShader("Texture", "diffuse", *color_texture)
    else:
        mat = newShader("Mesh", "diffuse", *color_mesh)
    bpy.ops.surface.primitive_nurbs_surface_sphere_add()
    bpy.context.active_object.data.materials.append(mat)
