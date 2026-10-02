from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

try:
    import OpenImageIO as oiio
except ImportError as error:
    raise RuntimeError("The Blender executable must include OpenImageIO") from error

SHADER = r"""shader ray_table_camera(
    string origin_map = "",
    string direction_map = "",
    int width = 1,
    output point position = point(0.0),
    output vector direction = vector(0.0),
    output color throughput = color(0.0))
{
    point raster = camera_shader_raster_position();
    float px = clamp(floor(raster[0] * width), 0.0, width - 1.0);
    float s = (px + 0.5) / width;
    float footprint = 0.0;
    float valid = 0.0;
    color origin = texture(origin_map, s, 0.5, 0.0, 0.0, 0.0, 0.0,
                           "interp", "closest", "wrap", "clamp", "alpha", footprint);
    color encoded_direction = texture(direction_map, s, 0.5, 0.0, 0.0, 0.0, 0.0,
                                      "interp", "closest", "wrap", "clamp", "alpha", valid);
    vector ray_direction = vector(encoded_direction[0], encoded_direction[1], encoded_direction[2]);
    if (valid > 0.5 && length(ray_direction) > 0.0) {
        position = point(origin[0], origin[1], origin[2]);
        direction = normalize(ray_direction);
        throughput = color(1.0);
    }
}"""


def write_exr(path: Path, pixels: np.ndarray) -> None:
    height, width, channels = pixels.shape
    spec = oiio.ImageSpec(width, height, channels, "float")
    output = oiio.ImageOutput.create(str(path))
    if output is None or not output.open(str(path), spec):
        raise RuntimeError(oiio.geterror())
    output.write_image(np.ascontiguousarray(pixels, dtype=np.float32))
    output.close()


def read_exr(path: Path) -> tuple[np.ndarray, list[str]]:
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise RuntimeError(oiio.geterror())
    names = list(image.spec().channelnames)
    pixels = image.read_image("float")
    image.close()
    return pixels, names


def render_pass(pixels: np.ndarray, names: list[str], components: str) -> np.ndarray:
    channels = [
        next(name for name in names if name == component or name.endswith(f".{component}"))
        for component in components
    ]
    return np.stack([pixels[..., names.index(channel)] for channel in channels], axis=-1)


def make_camera(shader: Path):
    camera = bpy.data.objects.get("RayTableCamera")
    if camera is None:
        camera_data = bpy.data.cameras.new("RayTableCamera")
        camera = bpy.data.objects.new("RayTableCamera", camera_data)
        bpy.context.scene.collection.objects.link(camera)
        camera.matrix_world = Matrix.Identity(4)
        camera.data.type = "CUSTOM"
        camera.data.custom_mode = "EXTERNAL"
        camera.data.clip_start = 1e-6
        camera.data.clip_end = 1e6
    camera.data.custom_filepath = str(shader)
    with bpy.context.temp_override(object=camera, active_object=camera):
        bpy.ops.object.camera_custom_update()
    return camera


def configure_cycles(scene, samples: int) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.shading_system = True
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_layers[0].use_pass_combined = True
    scene.view_layers[0].use_pass_position = True


def configure_camera(scene, renderer: str, samples: int) -> None:
    if renderer == "cycles":
        configure_cycles(scene, samples)
    elif renderer == "eevee":
        scene.render.engine = "BLENDER_EEVEE"
    elif renderer == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_shadows = True
    else:
        raise ValueError(f"Unsupported camera renderer: {renderer}")

    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "PNG"


def _material_info(material):
    color = np.asarray(material.diffuse_color[:], dtype=np.float32)
    image = None
    specular = np.full(3, 0.04, dtype=np.float32)
    shininess = 32.0
    if material.use_nodes:
        shader = next(
            (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        if shader is not None:
            base_color = shader.inputs.get("Base Color")
            if base_color is not None:
                color = np.asarray(base_color.default_value[:], dtype=np.float32)
                if base_color.is_linked and base_color.links[0].from_node.type == "TEX_IMAGE":
                    image = base_color.links[0].from_node.image
            metallic = float(shader.inputs.get("Metallic").default_value)
            roughness = float(shader.inputs.get("Roughness").default_value)
            ior = float(shader.inputs.get("IOR").default_value)
            specular_level = float(shader.inputs.get("Specular IOR Level").default_value)
            specular_tint = np.asarray(
                shader.inputs.get("Specular Tint").default_value[:3], dtype=np.float32
            )
            f0 = ((ior - 1) / (ior + 1)) ** 2 * 2 * specular_level
            tinted = f0 * ((1 - specular_tint) + specular_tint * color[:3])
            specular = (1 - metallic) * tinted + metallic * color[:3]
            shininess = 2 / max(roughness**4, 1e-4) - 2
    return color, image, specular, shininess


def _background_color(scene):
    world = scene.world
    if world is None:
        return np.zeros(3, dtype=np.float32)

    if world.use_nodes:
        background = next(
            (node for node in world.node_tree.nodes if node.type == "BACKGROUND"),
            None,
        )
        if background is not None:
            color = background.inputs.get("Color")
            if color is not None:
                return np.asarray(color.default_value[:3], dtype=np.float32)

    return np.asarray(world.color[:3], dtype=np.float32)


def extract_scene(scene, path: Path) -> None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    positions, normals, uvs, material_ids, mesh_ids = [], [], [], [], []
    materials, material_images, material_specular, material_shininess = [], [], [], []
    light_positions, light_directions, light_colors = [], [], []
    light_energies, light_types = [], []
    material_lookup = {}

    def get_material_id(material):
        key = material.as_pointer() if material is not None else None
        if key not in material_lookup:
            color, image, specular, shininess = (
                _material_info(material)
                if material is not None
                else (
                    np.array((0.8, 0.8, 0.8, 1), dtype=np.float32),
                    None,
                    np.full(3, 0.04, dtype=np.float32),
                    32.0,
                )
            )
            material_lookup[key] = len(materials)
            materials.append(color)
            material_images.append(image)
            material_specular.append(specular)
            material_shininess.append(shininess)
        return material_lookup[key]

    get_material_id(None)
    for mesh_index, object_ in enumerate(
        object_ for object_ in scene.objects if object_.type == "MESH"
    ):
        evaluated = object_.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            if not len(mesh.loop_triangles):
                continue
            vertex_positions = np.empty((len(mesh.vertices), 3), dtype=np.float32)
            corner_normals = np.empty((len(mesh.corner_normals), 3), dtype=np.float32)
            triangle_indices = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
            triangle_loops = np.empty_like(triangle_indices)
            mesh.vertices.foreach_get("co", vertex_positions.ravel())
            mesh.corner_normals.foreach_get("vector", corner_normals.ravel())
            mesh.loop_triangles.foreach_get("vertices", triangle_indices.ravel())
            mesh.loop_triangles.foreach_get("loops", triangle_loops.ravel())
            transform = np.asarray(object_.matrix_world, dtype=np.float32)
            linear = transform[:3, :3]
            positions.append(vertex_positions[triangle_indices] @ linear.T + transform[:3, 3])
            normal_matrix = np.linalg.inv(linear).T
            triangle_normals = corner_normals[triangle_loops] @ normal_matrix.T
            triangle_normals /= np.linalg.norm(triangle_normals, axis=-1, keepdims=True).clip(
                min=1e-8
            )
            normals.append(triangle_normals)
            uv_layer = mesh.uv_layers.active
            if uv_layer is None:
                uvs.append(np.zeros((len(triangle_indices), 3, 2), dtype=np.float32))
            else:
                loop_uvs = np.empty((len(mesh.loops), 2), dtype=np.float32)
                uv_layer.data.foreach_get("uv", loop_uvs.ravel())
                uvs.append(loop_uvs[triangle_loops])
            slots = [get_material_id(slot.material) for slot in object_.material_slots]
            triangle_materials = np.zeros(len(triangle_indices), dtype=np.int64)
            for index, triangle in enumerate(mesh.loop_triangles):
                if triangle.material_index < len(slots):
                    triangle_materials[index] = slots[triangle.material_index]
            material_ids.append(triangle_materials)
            mesh_ids.append(np.full(len(triangle_indices), mesh_index, dtype=np.int64))
        finally:
            evaluated.to_mesh_clear()
    if not positions:
        raise RuntimeError("glTF scene contains no mesh triangles")

    for object_ in (object_ for object_ in scene.objects if object_.type == "LIGHT"):
        transform = np.asarray(object_.matrix_world, dtype=np.float32)
        direction = transform[:3, :3] @ np.array((0, 0, 1), dtype=np.float32)
        direction /= np.linalg.norm(direction).clip(min=1e-8)
        light_positions.append(transform[:3, 3])
        light_directions.append(direction)
        light_colors.append(np.asarray(object_.data.color[:], dtype=np.float32))
        light_energies.append(float(object_.data.energy))
        light_types.append(0 if object_.data.type == "SUN" else 1)

    images = [image for image in material_images if image is not None]
    atlas_width = sum(int(image.size[0]) for image in images) or 1
    atlas_height = max((int(image.size[1]) for image in images), default=1)
    atlas = np.ones((atlas_height, atlas_width, 4), dtype=np.float32)
    texture_rects = np.full((len(materials), 4), -1, dtype=np.int32)
    image_offsets, x_offset = {}, 0
    for image in images:
        width, height = map(int, image.size)
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        atlas[:height, x_offset : x_offset + width] = np.flip(pixels, axis=0)
        image_offsets[image.as_pointer()] = (x_offset, 0, width, height)
        x_offset += width
    for index, image in enumerate(material_images):
        if image is not None:
            texture_rects[index] = image_offsets[image.as_pointer()]

    np.savez(
        path,
        background_color=_background_color(scene),
        positions=np.concatenate(positions, axis=0),
        normals=np.concatenate(normals, axis=0),
        uvs=np.concatenate(uvs, axis=0),
        material_id=np.concatenate(material_ids, axis=0),
        mesh_id=np.concatenate(mesh_ids, axis=0),
        material_color=np.stack(materials),
        material_specular=np.stack(material_specular),
        material_shininess=np.asarray(material_shininess, dtype=np.float32),
        texture_rect=texture_rects,
        texture_atlas=atlas,
        light_positions=np.asarray(light_positions, dtype=np.float32).reshape(-1, 3),
        light_directions=np.asarray(light_directions, dtype=np.float32).reshape(-1, 3),
        light_colors=np.asarray(light_colors, dtype=np.float32).reshape(-1, 3),
        light_energies=np.asarray(light_energies, dtype=np.float32),
        light_types=np.asarray(light_types, dtype=np.int64),
    )


def render_camera(scene, request: dict[str, object]) -> None:
    camera = bpy.data.objects.get("ViewerCamera")
    if camera is None:
        camera_data = bpy.data.cameras.new("ViewerCamera")
        camera = bpy.data.objects.new("ViewerCamera", camera_data)
        scene.collection.objects.link(camera)
    from mathutils import Matrix

    width, height = int(request["width"]), int(request["height"])
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36
    camera.data.lens = float(request["focal_length"][0]) * 36 / width
    rotation = Matrix(request["rotation"])
    camera.matrix_world = rotation.transposed().to_4x4()
    camera.location = request["position"]
    scene.camera = camera
    configure_camera(scene, str(request.get("renderer", "cycles")), int(request["samples"]))
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(request["output"])
    bpy.ops.render.render(write_still=True)


def setup_compositor(scene):
    scene.use_nodes = True
    node_tree = scene.compositing_node_group
    if node_tree is None:
        node_tree = bpy.data.node_groups.new("GaussianSplatsCompositor", "CompositorNodeTree")
        scene.compositing_node_group = node_tree
    node_tree.nodes.clear()
    layers = node_tree.nodes.new("CompositorNodeRLayers")
    combined = node_tree.nodes.new("CompositorNodeOutputFile")
    position = node_tree.nodes.new("CompositorNodeOutputFile")
    combined.file_output_items.new("RGBA", "combined")
    position.file_output_items.new("RGBA", "position")
    node_tree.links.new(layers.outputs["Image"], combined.inputs[0])
    node_tree.links.new(layers.outputs["Position"], position.inputs[0])
    return combined, position


def query(scene, origins, directions, footprints, workdir, request, samples, batch_size):
    configure_cycles(scene, samples)
    colors = np.zeros((len(origins), 3), dtype=np.float32)
    alphas = np.zeros(len(origins), dtype=np.float32)
    hits = np.zeros((len(origins), 3), dtype=np.float32)
    combined_output, position_output = setup_compositor(scene)
    for batch, start in enumerate(range(0, len(origins), batch_size)):
        stop = min(start + batch_size, len(origins))
        width = stop - start
        render_width = max(width, 4)
        origin_table = np.zeros((1, render_width, 4), dtype=np.float32)
        direction_table = np.zeros((1, render_width, 4), dtype=np.float32)
        origin_table[0, :width, :3] = origins[start:stop]
        origin_table[0, :width, 2] *= -1
        origin_table[0, :width, 3] = footprints[start:stop]
        direction_table[0, :width, :3] = -directions[start:stop]
        direction_table[0, :width, 2] *= -1
        direction_table[0, :width, 3] = 1
        stem = f"{request}-{batch:05d}"
        origin_path = workdir / f"{stem}.origins.exr"
        direction_path = workdir / f"{stem}.directions.exr"
        shader = workdir / f"{stem}.osl"
        write_exr(origin_path, origin_table)
        write_exr(direction_path, direction_table)
        shader.write_text(
            SHADER.replace('string origin_map = ""', f'string origin_map = "{origin_path}"')
            .replace('string direction_map = ""', f'string direction_map = "{direction_path}"')
            .replace("int width = 1", f"int width = {render_width}")
        )
        scene.camera = make_camera(shader)
        scene.render.resolution_x, scene.render.resolution_y = render_width, 1
        for output, suffix in ((combined_output, "combined"), (position_output, "position")):
            output.directory = str(workdir)
            output.file_name = f"{stem}.{suffix}"
            output.format.file_format = "OPEN_EXR_MULTILAYER"
            output.format.color_depth = "32"
        bpy.ops.render.render()
        combined_path = next(workdir.glob(f"{stem}.combined*.exr"))
        position_path = next(workdir.glob(f"{stem}.position*.exr"))
        combined, _ = read_exr(combined_path)
        position, _ = read_exr(position_path)
        combined = render_pass(combined, list(_), "RGBA").reshape(-1, 4)[:width]
        position = position[..., :3].reshape(-1, 3)[:width]
        colors[start:stop], alphas[start:stop] = combined[:, :3], combined[:, 3]
        hits[start:stop] = position
    np.savez(workdir / f"{request}.result.npz", colors=colors, alphas=alphas, hits=hits)


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    scene_path, scene_output = arguments[:2]
    server = "--server" in arguments
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(Path(scene_path).resolve()))
    scene = bpy.context.scene
    extract_scene(scene, Path(scene_output))
    print("READY", flush=True)
    if not server:
        raise RuntimeError("The Blender worker must run in server mode")
    for line in sys.stdin:
        request = json.loads(line)
        if request["command"] == "close":
            break
        if request["command"] == "query":
            input_path = Path(request["input"])
            output_path = Path(request["output"])
            data = np.load(input_path)
            query(
                scene,
                data["origins"],
                data["directions"],
                data["footprints"],
                input_path.parent,
                output_path.name.removesuffix(".result.npz"),
                int(request["samples"]),
                int(request["batch_size"]),
            )
            print(f"RESULT {output_path}", flush=True)
        elif request["command"] == "render":
            render_camera(scene, request)
            print(f"RENDER_RESULT {request['output']}", flush=True)


if __name__ == "__main__":
    main()
