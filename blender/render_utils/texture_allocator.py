
class TextureAllocator:

    def __init__(self, bpy, texture_name='texture_material'):
        self.bpy = bpy
        self.texture_name = texture_name
        # self.init_texture()

    def init_texture(self):
        bpy = self.bpy
        texture_name = self.texture_name
        mat = bpy.data.materials.new(name=texture_name)
        mat.use_nodes = True
        if mat.node_tree:
            mat.node_tree.links.clear()
            mat.node_tree.nodes.clear()
        
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        output = nodes.new(type='ShaderNodeOutputMaterial')
        # shader = nodes.new(type='ShaderNodeBsdfDiffuse')
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        links.new(shader.outputs[0], output.inputs[0])

        input_attribute = nodes.new(type='ShaderNodeAttribute')
        input_attribute.attribute_name = 'Col'
        links.new(input_attribute.outputs[0], shader.inputs[0])
        # return mat

    def set_texture(self):
        bpy = self.bpy  
        bpy.context.active_object.data.materials.append(bpy.data.materials[self.texture_name])
