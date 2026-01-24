"""Domain package for paired-assets analysis.

Important: this package must stay lightweight to avoid circular imports.

Rule of thumb:
- `ports` may import `domain.model` for type hints.
- `domain.__init__` must NOT eagerly import `domain.services` / `domain.strategies`,
  because those modules import `ports` (creating a cycle).
"""

# Re-export only the stable domain model types here.
# Import services/strategies directly from their modules when needed:
#   from paired_assets_analysis.domain.services import SceneAnalysisService
#   from paired_assets_analysis.domain.strategies import BackendPatchworkPPSurfaceEstimator

from .model import *  # noqa: F403
