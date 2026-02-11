---
title: "I Built a 2D → 3D Generator on a 4GB GPU (And Patched Half the Stack to Make It Work)"
date: 2026-02-12
id: 11
author: "Afnan K Salal"
authorGithub: "https://github.com/afnanksalal"
tags:
  - AI
  - 3D Reconstruction
  - PyTorch
  - CUDA
  - TripoSR
  - SAM2
  - Multi View Reconstruction
  - Windows
  - Systems Engineering
  - GPU Optimization
---

# This Was Supposed to Be Simple.

The idea sounded clean.

Take an image → segment object → estimate depth → reconstruct 3D → export mesh.

Elegant.

Minimal.

Research-paper vibes.

Instead, it became:

- 3 days of compilation failures  
- Internal PyTorch CMake patching  
- CUDA version pinning like a landmine  
- Modifying TripoSR internals  
- Breaking and reassembling multi-view mesh fusion  
- Aggressively re-engineering GPU memory flow  
- Listening to my RTX 3050 beg for mercy  

And yes.

It works.

On 4GB VRAM.

# The Goal

Build a production-ready 2D → 3D reconstruction pipeline that:

- Works on consumer hardware  
- Handles single-view and multi-view  
- Produces clean meshes  
- Doesn’t OOM instantly  
- Actually aligns multi-view geometry properly  

Stack:

- **SAM 2.1** – segmentation  
- **Depth Anything V2** – monocular depth  
- **TripoSR** – neural surface reconstruction  
- **PyTorch 2.6.0 + CUDA 12.4**  
- **Open3D + Trimesh** for alignment  
- Custom memory management layer  
- Gradio UI  

Simple architecture.

Terrifying implementation.

# Day 1 – The Binary Compatibility Trap

TripoSR requires `torchmcubes`.

Installed it.

Instant failure:

DLL load failed while importing torchmcubes_module

Why?

Because prebuilt wheels were compiled for older PyTorch builds.

I was running:

- PyTorch 2.6.0
- CUDA 12.4

The wheel expected something else.

Binary mismatch.

So I compiled from source.

Which triggered the real chaos.

# CUDA Version Hell (Yes, It Matters That Much)

PyTorch 2.6.0 **requires CUDA 12.4**.

Not “close enough”.
Not “newer should work”.

Exactly 12.4.

I had 12.1.

That alone was enough to destroy everything.

Reinstalled CUDA.

Reinstalled PyTorch.

Rebuilt environment.

4 hours gone.

# nvToolsExt – The Ghost of CUDA Past

Compilation failed with:

CUDA::nvToolsExt target not found

Why?

CUDA 12 removed nvToolsExt from toolkit.

But PyTorch’s CMake config still references it.

So CMake is looking for something that literally no longer exists.

So what did I do?

I patched PyTorch’s internal:

torch/share/cmake/Caffe2/public/cuda.cmake

Commented out the `CUDA::nvToolsExt` target dependency.

Yes.

I modified framework-level build configuration.

Then cloned NVTX manually just for header resolution.

Recompiled.

It worked.

At this point I stopped pretending this was a normal install.

# Windows + C++ Toolchain Roulette

CMake couldn’t find:

- cl.exe  
- rc.exe  
- kernel32.lib  

Visual Studio Community?
Nope.

Visual Studio 2019?
Nope.

Only **Visual Studio 2022 Build Tools** worked reliably.

And even then, I had to manually set:

- PATH  
- LIB  
- INCLUDE  
- CUDA_PATH  

Like it was 2008.

# Day 2 – It Compiles. Now It Crashes.

torchmcubes finally compiled.

TripoSR started running.

Then VRAM exploded.

SAM2 alone can eat ~2GB.

TripoSR grid sampling spikes memory.

On a 4GB GPU?

That’s instant death.

So I redesigned execution flow.

# The Memory Architecture Rewrite

Instead of:

All models on GPU → run sequentially

I built:

Stage-based GPU ownership.

Pipeline:

1. Depth (CPU only)
2. Segmentation (CPU only)
3. Offload everything
4. Nuclear CUDA reset
5. Move only TripoSR to GPU
6. Run in FP16
7. Offload again

Added:

```python
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()
torch.cuda.synchronize()
````

Multiple times.

Aggressively.

Peak VRAM:

1.7GB

Idle:

0.14GB

I basically treated VRAM like a shared apartment with eviction notices.

# The Segmentation Problem

Naively prompting SAM2 with a single center point?

Terrible results.

Instead I built depth-guided prompting:

1. Generate depth map
2. Compute 70th percentile threshold
3. Extract closest region
4. Sample multiple foreground points
5. Generate bounding box
6. Feed structured prompts to SAM2

Segmentation accuracy jumped massively.

Depth turned segmentation from guessing into guided inference.

This alone improved final mesh quality.

# The Real Surgery: TripoSR Modifications

Now the fun part.

TripoSR works fine for single view.

Multi-view?

Absolute chaos.

Meshes misaligned.

Density thresholds inconsistent.

Artifacts everywhere.

So I modified internal behavior.

### 1. Adaptive Marching Cubes Threshold

Default extraction uses fixed density threshold.

That fails when multi-view density distributions differ.

I changed it to:

* Query entire density grid
* Compute median density
* Clamp between safe bounds
* Use adaptive threshold

This reduced holes and floating artifacts significantly.

### 2. Multi-View Mesh Alignment Rewrite

Default merging? Basically stacking geometry.

Nope.

Implemented:

* View-aware rotation presets (Front/Back/Left/Right)
* FPFH feature extraction
* Global registration
* ICP refinement
* Outlier rejection pass

Using:

* Open3D
* SciPy
* Trimesh

Now meshes actually align in 3D space instead of becoming a polygon soup.

### 3. GPU-Safe Grid Resolution Handling

Resolution scaling was naive.

At 256 resolution:

OOM 50% of the time.

At 512?

Immediate crash.

So I implemented:

Resolution fallback logic:

```python
for resolution in [512, 256, 128]:
    try:
        run_inference(resolution)
        break
    except RuntimeError:
        torch.cuda.empty_cache()
```

On 4GB?

Resolution 128 is reality.

It looks like GTA 3.

But it works.

# The NV-Level Tweaks

Beyond CMake patching, I had to:

* Ensure correct CUDA compute capability flags
* Avoid unnecessary debug instrumentation
* Force FP16 inference path
* Avoid mixed tensor precision mismatches

Because one stray FP32 allocation during grid sampling?

OOM.

# Performance (RTX 3050 4GB)

Single View:

* Depth: ~7s (CPU)
* Segmentation: ~13s (CPU)
* 3D Reconstruction: ~11s (GPU)
* Total: ~35–40 seconds

Multi View (4 views):

* ~2.5 minutes
* Alignment adds ~15s

Mesh:

* 128 resolution
* ~17k vertices
* ~34k faces
* Geometrically correct
* Visually low-poly

But stable.

# Screenshot Section

### Depth Map

*(Insert depth map screenshot)*

### Segmentation Mask

*(Insert segmentation screenshot)*

### Raw Multi-View Mesh Before Alignment

*(Insert screenshot)*

### Final Aligned Mesh

*(Insert screenshot)*

# What This Actually Taught Me

1. ML engineering is 50% systems engineering.
2. Version compatibility matters more than model architecture.
3. GPU memory management is architecture design.
4. Research code is not production-ready.
5. Consumer hardware forces better engineering.

And most importantly:

You don’t need a 4090 to build serious systems.

You need:

* patience
* willingness to patch internals
* and acceptance that your GPU will sound like an aircraft turbine.

# The Final Pipeline

```
Image
 ↓
Depth Anything V2 (CPU)
 ↓
SAM 2.1 (Depth-Guided Prompting)
 ↓
Full Memory Cleanup
 ↓
Enhanced TripoSR (FP16, Adaptive Threshold)
 ↓
Multi-View Registration (FPFH + ICP)
 ↓
GLB / OBJ Export
```

# Conclusion

Building this system wasn’t about using big models.

It was about bending the entire stack to fit inside 4GB VRAM.

I patched CUDA configs.
I modified internal SR behavior.
I redesigned memory flow.
I re-engineered multi-view fusion.

And it works.

Does it look like Unreal Engine 5?

No.

It looks like PS2-era geometry.

But the math is correct.
The topology is stable.
The alignment works.

And it runs on a laptop GPU.

That’s engineering.

And honestly?

I’m more proud of making it work on 4GB
than I would be running it effortlessly on 24GB.

Constraints create real skill.

---
