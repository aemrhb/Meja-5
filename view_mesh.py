import argparse
import os
import sys

import trimesh


def view_mesh(mesh_path: str, cull: bool = False, smooth: bool = True, wireframe: bool = False) -> None:
    if not os.path.isfile(mesh_path):
        print(f"File not found: {mesh_path}")
        sys.exit(1)

    try:
        mesh = trimesh.load(mesh_path, force='mesh')
    except Exception as exc:
        print(f"Failed to load mesh: {exc}")
        sys.exit(1)

    if mesh.is_empty:
        print("Loaded geometry is empty or not a mesh.")
        sys.exit(1)

    # Configure visual settings
    if smooth and hasattr(mesh, 'vertex_normals'):
        # Ensure normals are present for nicer shading
        try:
            mesh.compute_vertex_normals()
        except Exception:
            pass

    if wireframe:
        # Create scene with edges emphasized
        scene = trimesh.Scene()
        scene.add_geometry(mesh)
        try:
            scene.show(flags={'wireframe': True, 'cull': cull})
        except BaseException as exc:
            print("Viewer failed to open. If you're on a headless server, ensure you have a display (e.g., via X11 or Xvfb).")
            print(f"Error: {exc}")
            sys.exit(2)
        return

    # Default: shaded view, keep back-face culling configurable
    try:
        # Using Scene gives a bit more consistent behavior across backends
        scene = trimesh.Scene(mesh)
        scene.show(flags={'cull': cull})
    except BaseException as exc:
        print("Viewer failed to open. If you're on a headless server, ensure you have a display (e.g., via X11 or Xvfb).")
        print(f"Error: {exc}")
        sys.exit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open and visualize a saved mesh (e.g., PCA-colored PLY) using trimesh.")
    parser.add_argument("mesh", type=str, help="Path to mesh file (.ply, .obj, .off, etc.)")
    parser.add_argument("--no_cull", action="store_true", help="Disable back-face culling in the viewer.")
    parser.add_argument("--flat", action="store_true", help="Use flat shading (disable smoothing).")
    parser.add_argument("--wireframe", action="store_true", help="Render as wireframe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    view_mesh(
        mesh_path=args.mesh,
        cull=not args.no_cull,
        smooth=not args.flat,
        wireframe=args.wireframe,
    )


if __name__ == "__main__":
    main()


