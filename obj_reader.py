import pywavefront as wave
scene = wave.Wavefront('./pcd files/mug.obj')
for name, material in scene.materials.items():
    print(name)
    print(material.vertices)