import io

import trimesh
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, FileResponse

from services.generator_registry import WORKSPACE_DIR

router = APIRouter(tags=["export"])

SUPPORTED = {"glb", "stl", "obj", "ply", "3mf"}


@router.get("/{fmt}")
def export_mesh(fmt: str, path: str):
    if fmt not in SUPPORTED:
        raise HTTPException(400, f"Unsupported format: {fmt}. Supported: {', '.join(SUPPORTED)}")

    full_path = (WORKSPACE_DIR / path).resolve()
    if not str(full_path).startswith(str(WORKSPACE_DIR.resolve())):
        raise HTTPException(400, "Invalid path")
    if not full_path.exists():
        raise HTTPException(404, f"File not found: {path}")

    # GLB — serve directly, no conversion needed
    if fmt == "glb":
        return FileResponse(str(full_path), media_type="model/gltf-binary")

    # Load and flatten scene to a single mesh
    loaded = trimesh.load(str(full_path))
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    else:
        mesh = loaded

    # 3MF — multi-colour export for Bambu Lab / AMS printers
    if fmt == "3mf":
        from services.threemf_exporter import export_3mf
        data = export_3mf(mesh)
        return Response(
            content=data,
            media_type="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
            headers={"Content-Disposition": "attachment; filename=model.3mf"},
        )

    buf = io.BytesIO()
    if fmt == "stl":
        mesh.export(buf, file_type="stl")
        media_type = "model/stl"
    elif fmt == "ply":
        mesh.export(buf, file_type="ply")
        media_type = "application/octet-stream"
    else:  # obj
        mesh.export(buf, file_type="obj")
        media_type = "text/plain"

    buf.seek(0)
    return Response(content=buf.read(), media_type=media_type)
