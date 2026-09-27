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


def extract_scene(scene, path: Path) -> None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    positions, normals, mesh_ids = [], [], []
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
            vertex_normals = np.empty_like(vertex_positions)
            triangle_indices = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
            mesh.vertices.foreach_get("co", vertex_positions.ravel())
            mesh.vertices.foreach_get("normal", vertex_normals.ravel())
            mesh.loop_triangles.foreach_get("vertices", triangle_indices.ravel())
            transform = np.asarray(object_.matrix_world, dtype=np.float32)
            linear = transform[:3, :3]
            positions.append(vertex_positions[triangle_indices] @ linear.T + transform[:3, 3])
            normal_matrix = np.linalg.inv(linear).T
            triangle_normals = vertex_normals[triangle_indices] @ normal_matrix.T
            triangle_normals /= np.linalg.norm(triangle_normals, axis=-1, keepdims=True).clip(
                min=1e-8
            )
            normals.append(triangle_normals)
            mesh_ids.append(np.full(len(triangle_indices), mesh_index, dtype=np.int64))
        finally:
            evaluated.to_mesh_clear()
    if not positions:
        raise RuntimeError("glTF scene contains no mesh triangles")
    np.savez(
        path,
        positions=np.concatenate(positions, axis=0),
        normals=np.concatenate(normals, axis=0),
        mesh_id=np.concatenate(mesh_ids, axis=0),
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
    camera.matrix_world = (
        rotation.transposed() @ Matrix(((1, 0, 0), (0, 1, 0), (0, 0, -1)))
    ).to_4x4()
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
    scene_path, samples, batch_size, scene_output = arguments[:4]
    samples, batch_size = int(samples), int(batch_size)
    server = len(arguments) > 4 and arguments[4] == "--server"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(Path(scene_path).resolve()))
    scene = bpy.context.scene
    configure_cycles(scene, samples)
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
                samples,
                batch_size,
            )
            print(f"RESULT {output_path}", flush=True)
        elif request["command"] == "render":
            request["samples"] = samples
            render_camera(scene, request)
            print(f"RENDER_RESULT {request['output']}", flush=True)


if __name__ == "__main__":
    main()
