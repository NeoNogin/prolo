"""
Export a trimesh mesh as a multi-color .3mf file.

Quantises the mesh's texture (or vertex colours) down to *num_colors*
distinct materials and assigns every triangle to one of them.  The
resulting file is compatible with Bambu Studio and other slicers that
support the 3MF Materials & Properties Extension.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_3mf(mesh: trimesh.Trimesh, num_colors: int = 4) -> bytes:
    """Return the raw bytes of a .3mf ZIP archive for *mesh*.

    Every triangle is assigned to one of *num_colors* materials whose
    colours are derived from the mesh's visual data (texture or vertex
    colours).  If the mesh carries no colour information a single grey
    material is used instead.
    """
    face_colors = _extract_face_colors(mesh)
    palette, labels = _quantize_colors(face_colors, num_colors)
    return _build_3mf(mesh, palette, labels)


# ---------------------------------------------------------------------------
# Step 1 – per-face colour extraction
# ---------------------------------------------------------------------------

def _extract_face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return an (F, 3) uint8 RGB array – one colour per face."""

    visual = mesh.visual

    # --- Textured mesh (UV + image) ---
    if hasattr(visual, "uv") and visual.uv is not None:
        try:
            mat = visual.material
            image = None
            # SimpleMaterial / PBRMaterial may store the texture differently
            if hasattr(mat, "image") and mat.image is not None:
                image = mat.image
            elif hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                image = mat.baseColorTexture
            if image is not None:
                return _sample_texture(mesh, visual.uv, image)
        except Exception:
            pass  # fall through to vertex colours or default

    # --- Vertex-colour mesh ---
    if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
        vc = np.asarray(visual.vertex_colors)  # (V, 4) RGBA uint8
        if vc.ndim == 2 and vc.shape[0] == len(mesh.vertices):
            face_vc = vc[mesh.faces]  # (F, 3, 4)
            return face_vc[:, :, :3].mean(axis=1).astype(np.uint8)

    # --- Face colours (trimesh sometimes stores these directly) ---
    if hasattr(visual, "face_colors") and visual.face_colors is not None:
        fc = np.asarray(visual.face_colors)
        if fc.ndim == 2 and fc.shape[0] == len(mesh.faces):
            return fc[:, :3].astype(np.uint8)

    # --- Fallback: uniform grey ---
    return np.full((len(mesh.faces), 3), 160, dtype=np.uint8)


def _sample_texture(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    image,
) -> np.ndarray:
    """Sample the texture image at each face's centroid UV coordinate."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(io.BytesIO(image))
    image = image.convert("RGB")
    width, height = image.size
    pixels = np.asarray(image)  # (H, W, 3) uint8

    # Centroid UV for each face: average of the 3 vertex UVs
    face_uv = uv[mesh.faces]  # (F, 3, 2)
    centroid_uv = face_uv.mean(axis=1)  # (F, 2)

    # Wrap to [0, 1) and convert to pixel coords
    u = centroid_uv[:, 0] % 1.0
    v = centroid_uv[:, 1] % 1.0

    px = np.clip((u * width).astype(int), 0, width - 1)
    py = np.clip(((1.0 - v) * height).astype(int), 0, height - 1)  # Y-flip

    return pixels[py, px]  # (F, 3) uint8


# ---------------------------------------------------------------------------
# Step 2 – colour quantisation (k-means)
# ---------------------------------------------------------------------------

def _quantize_colors(
    face_colors: np.ndarray, num_colors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (palette, labels).

    *palette* is (K, 3) uint8 – the K representative colours.
    *labels* is (F,) int   – per-face index into the palette.
    """
    unique = np.unique(face_colors, axis=0)
    k = min(num_colors, len(unique))

    if k <= 1:
        palette = unique[:1] if len(unique) else face_colors[:1]
        labels = np.zeros(len(face_colors), dtype=int)
        return palette.astype(np.uint8), labels

    # Attempt scipy first; fall back to a simple numpy implementation.
    try:
        from scipy.cluster.vq import kmeans2
        obs = face_colors.astype(np.float64)
        centroids, labels = kmeans2(obs, k, minit="points", iter=20)
    except ImportError:
        centroids, labels = _numpy_kmeans(face_colors.astype(np.float64), k)

    palette = np.clip(centroids, 0, 255).astype(np.uint8)
    return palette, labels


def _numpy_kmeans(
    data: np.ndarray, k: int, max_iter: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal k-means using only numpy (fallback when scipy is absent)."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(data), size=k, replace=False)
    centroids = data[indices].copy()

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        # Recompute centroids
        new_centroids = np.empty_like(centroids)
        for j in range(k):
            members = data[labels == j]
            new_centroids[j] = members.mean(axis=0) if len(members) else centroids[j]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# ---------------------------------------------------------------------------
# Step 3 – 3MF archive construction
# ---------------------------------------------------------------------------

_NS_3MF = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"

_CONTENT_TYPES_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

_RELS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def _color_hex(rgb: np.ndarray) -> str:
    return "#{:02X}{:02X}{:02X}FF".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _build_3mf(
    mesh: trimesh.Trimesh,
    palette: np.ndarray,
    labels: np.ndarray,
) -> bytes:
    """Assemble the .3mf ZIP archive.

    Builds the 3dmodel.model XML by hand (no ElementTree) so we have
    exact control over namespace declarations and attribute order, which
    matters for Bambu Studio's parser.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    parts: list[str] = []
    w = parts.append

    w('<?xml version="1.0" encoding="UTF-8"?>')
    w('<model unit="millimeter" xml:lang="en-US" '
      'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
      'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">')
    w('<resources>')

    # colorgroup — one <m:color> per palette colour (Bambu Studio requires
    # m:colorgroup, it ignores core basematerials)
    w('<m:colorgroup id="1">')
    for rgb in palette:
        w(f'<m:color color="{_color_hex(rgb)}" />')
    w('</m:colorgroup>')

    # object with default material (pid/pindex required by spec)
    w('<object id="2" type="model" pid="1" pindex="0">')
    w('<mesh>')

    # vertices
    w('<vertices>')
    for v in vertices:
        w(f'<vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}" />')
    w('</vertices>')

    # triangles with per-face material
    w('<triangles>')
    for i, face in enumerate(faces):
        w(f'<triangle v1="{int(face[0])}" v2="{int(face[1])}" '
          f'v3="{int(face[2])}" pid="1" p1="{int(labels[i])}" />')
    w('</triangles>')

    w('</mesh>')
    w('</object>')
    w('</resources>')
    w('<build>')
    w('<item objectid="2" />')
    w('</build>')
    w('</model>')

    model_xml = "\n".join(parts).encode("utf-8")

    # -- Pack into ZIP ----------------------------------------------------
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", _RELS_XML)
        zf.writestr("3D/3dmodel.model", model_xml)

    return buf.getvalue()
