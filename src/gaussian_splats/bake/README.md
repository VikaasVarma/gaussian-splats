# Mesh-to-splat bake

Convert a glTF/GLB mesh's geometry and appearance into Gaussian splats using a rendering engine.


## Pipeline

1. Sample splats on the mesh surface and align them to its normals.
2. Query radiance in several outgoing directions per splat.
3. Fit View-Dependent SH color opacity.
4. Post-process (optional SH smoothing) and checkpoint


## Usage

The below bakes the Cornell box using blender's cycles renderer.

```bash
uv run python -m gaussian_splats.bake bake cycles \
  --mesh assets/cornell-box/cornell_box_core.glb \
  --output output/cornell.pt \
  --cycles-device OPTIX \
  --resolution 256 --sigma 0.6825 \
  --samples 128 --view-samples 128 --sh-degree 8 \
  --exposure 4 --sh-smoothing-percent 10 \
  --cycles-workers 2
```

View the checkpoint:

```bash
uv run visualize
```

Open <http://127.0.0.1:7007/gaussian/> and select `output/cornell.pt`.
See the [viewer README](../visualize/README.md) for scene previews and renderer comparisons.

## Examples

| Scene | Renderer | Splats | Camera Rays (M) | Bake Rays (M) | SSIM | PSNR (dB) |
|---|---|---:|---:|---:|---:|---:|
| Cornell box | Cycles | 425,242 | 1073.74 | 54.43 | 0.9704 | 35.42 |
| Damaged Helmet | Cycles | 250,000 | 1073.74 | 32.00 | 0.9570 | 31.10 |
| Avocado | Cycles | 250,000 | 1073.74 | 32.00 | 0.9890 | 42.26 |
| Duck | Phong | 250,000 | 0.26 | 8.00 | 0.9641 | 25.69 |
| Barramundi Fish | Phong | 250,000 | 0.26 | 8.00 | 0.9326 | 32.85 |

Camera rays are primary samples per 512×512 view; bake rays are outgoing radiance queries across all splats, excluding path samples and secondary/shadow rays. Image stats average four full-frame 8-bit sRGB views.

Cycles: SH8, 128 directions, covariance epsilon 0.03. Phong: SH2, 32 directions, epsilon 0.3. No refinement.

### Cornell box · Cycles

![Four Cornell box views: Cycles left, Gaussian splats right](examples/cornell.png)

### Damaged Helmet · Cycles

![Four Damaged Helmet views: Cycles left, Gaussian splats right](examples/cycles-damaged-helmet.png)

### Avocado · Cycles

![Four Avocado views: Cycles left, Gaussian splats right](examples/cycles-avocado.png)

### Duck · Phong

![Four Duck views: Phong left, Gaussian splats right](examples/phong-duck.png)

### Barramundi Fish · Phong

![Four Barramundi Fish views: Phong left, Gaussian splats right](examples/phong-fish.png)


## Credits

Inspiration and thanks to [Mesh2Splat](https://github.com/electronicarts/mesh2splat).

### Asset credits

| Asset | Credits and license |
|---|---|
| [Damaged Helmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DamagedHelmet) | ctxwing (2018), CC BY 4.0; earlier model by theblueturtle_ (2016), CC BY-NC 4.0 |
| [Boom Box](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoomBox) | Microsoft (2017), CC0 1.0 |
| [Avocado](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Avocado) | Microsoft (2017), CC0 1.0 |
| [Barramundi Fish](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BarramundiFish) | Microsoft (2017), CC0 1.0 |
| [Duck](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Duck) | Sony (2006), SCEA Shared Source License 1.0 |
| [Fox](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox) | PixelMannen model (2014), CC0 1.0; tomkranis rigging/animation (2014), CC BY 4.0; @AsoboStudio and @scurest glTF conversion (2017), CC BY 4.0 |
