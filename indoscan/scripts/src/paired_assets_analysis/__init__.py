"""Paired assets analysis bounded context (DDD layered package).

This package intentionally keeps `__init__` **side-effect free**: importing the
package should not import heavy numeric libraries or IO backends.

Use explicit imports for entrypoints:
`from paired_assets_analysis.entrypoints.paired_assets_analyze import paired_assets_analyze`
"""

__all__: list[str] = []
