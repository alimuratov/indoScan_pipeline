"""Infrastructure layer: adapters for paired-assets analysis.

Keep this package side-effect free: importing `paired_assets_analysis.infrastructure`
should not import heavy numeric libraries (NumPy/Open3D/Patchwork++).

Import adapters explicitly from their modules, e.g.:
- `from paired_assets_analysis.infrastructure.filesystem import FilesystemRoadSceneSource`
- `from paired_assets_analysis.infrastructure.pcdtools_adapter import Open3DPointCloudFormatHandler`
"""

__all__: list[str] = []
