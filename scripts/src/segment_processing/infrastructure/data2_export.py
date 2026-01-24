"""Export segment artifacts to legacy-compatible output/data2 target tree.

This module implements a SegmentArtifactsRepository that writes a target tree
compatible with the legacy `copy_roads_to_target.py` script:

<export_root>/
  copy_manifest.json
  Roads/
    rd-<uuid>/
      rs-<uuid>/
        Data/
          Survey video/
            output_video.mp4
          Lidar Scan/
            <segment_scan>.pcd
        imu.json
        segment_depth_timestamps.json
        segment_gps.json
        segment_meta.json
        Potholes/
          pt-<uuid>/
            Image/<timestamp>.jpg
            Lidar Scan/<timestamp>.pcd
            pothole_meta.json

IDs (rd/rs/pt) are generated using the same prefix + uuid4 strategy as legacy.
We persist them in copy_manifest.json so repeated exports for the same sources
reuse the same IDs.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from segment_processing.domain.model import SegmentProcessingResult


def _generate_prefixed_uuid(prefix: str) -> str:
    """Generate a random id string with a given prefix, e.g., 'rd-<uuid>'."""
    import uuid
    return f"{prefix}-{uuid.uuid4()}"


logger = logging.getLogger(__name__)


@dataclass
class Data2SegmentArtifactsRepository:
    """Export a processed segment to a legacy-compatible `output/data2` tree."""

    export_root: Path
    roads_dir_name: str = "Roads"
    manifest_filename: str = "copy_manifest.json"
    pothole_prefix: str = "pothole_"

    # --------------------------- internal helpers --------------------------- #

    @staticmethod
    def _write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=indent) +
                        "\n", encoding="utf-8")

    @staticmethod
    def _copy_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def _find_first_immediate_with_suffixes(
        folder: Path,
        suffixes: Tuple[str, ...],
    ) -> Optional[Path]:
        if not folder.is_dir():
            return None
        suffix_set = {s.lower() for s in suffixes}
        files = sorted(p for p in folder.iterdir() if p.is_file()
                       and p.suffix.lower() in suffix_set)
        return files[0] if files else None

    def _discover_pothole_dirs(self, segment_dir: Path) -> List[Path]:
        if not segment_dir.is_dir():
            return []
        return sorted(
            p for p in segment_dir.iterdir() if p.is_dir() and p.name.startswith(self.pothole_prefix)
        )

    def _find_pothole_media(self, pothole_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        img = self._find_first_immediate_with_suffixes(
            pothole_dir, (".jpg", ".png"))
        pcd = self._find_first_immediate_with_suffixes(pothole_dir, (".pcd",))
        return img, pcd

    @staticmethod
    def _extract_pothole_meta_from_inspection(inspection_json: Path) -> Optional[Dict[str, float]]:
        try:
            data = json.loads(inspection_json.read_text(encoding="utf-8"))
        except Exception:
            return None

        metadata = data.get("metadata") if isinstance(data, dict) else None
        if not isinstance(metadata, dict):
            return None

        best_metrics: Dict[str, Any] | None = None

        # Preferred shape for per-pothole folders: metadata.metrics
        m = metadata.get("metrics")
        if isinstance(m, dict):
            best_metrics = m
        else:
            # Multi-cluster shape (if any): pick the deepest cluster metrics.
            clusters = metadata.get("clusters")
            if isinstance(clusters, list) and clusters:
                best_depth = float("-inf")
                for c in clusters:
                    if not isinstance(c, dict):
                        continue
                    cm = c.get("metrics")
                    if not isinstance(cm, dict):
                        continue
                    try:
                        d = float(cm.get("max_depth"))
                    except Exception:
                        continue
                    if d > best_depth:
                        best_depth = d
                        best_metrics = cm

        if best_metrics is None:
            return None

        try:
            depth = float(best_metrics.get("max_depth", 0.0) or 0.0)
        except Exception:
            depth = 0.0

        try:
            area = float(best_metrics.get("hull_area", 0.0) or 0.0)
        except Exception:
            area = 0.0

        # Prefer convex volume; fallback to delaunay.
        vol_val = best_metrics.get("volume_convex", None)
        if vol_val is None:
            vol_val = best_metrics.get("volume_delaunay", None)
        try:
            volume = float(vol_val or 0.0)
        except Exception:
            volume = 0.0

        return {"depth": depth, "volume": volume, "area": area}

    def _extract_pothole_meta(self, pothole_dir: Path) -> Optional[Dict[str, float]]:
        """Extract normalized pothole meta for legacy consumers.

        Prefer `inspection_result.json` (new).
        """
        inspection = pothole_dir / "inspection_result.json"
        if inspection.is_file():
            meta = self._extract_pothole_meta_from_inspection(inspection)
            if meta is not None:
                return meta

        return None

    def save(self, *, result: SegmentProcessingResult) -> None:
        export_root = self.export_root.resolve()
        roads_root = export_root / self.roads_dir_name
        manifest_path = export_root / self.manifest_filename

        # determine which road this segment belongs to
        segment_source_dir = result.segment.source_dir.resolve()
        road_source_dir = segment_source_dir.parent.resolve()

        roads_root.mkdir(parents=True, exist_ok=True)

        manifest = self._load_or_init_manifest_v2(
            manifest_path=manifest_path,
            roads_root_guess=road_source_dir.parent,
            target_roads_dir=roads_root,
        )

        road_source_str = str(road_source_dir)
        segment_source_str = str(segment_source_dir)

        # --- O(1) manifest lookups (keyed by source paths) ---
        roads_by_source, segments_by_source = self.manifest_lookups(manifest)
        segment_entry, segment_target_dir = self.ensure_output_locations(
            roads_root=roads_root,
            road_source_str=road_source_str,
            segment_source_str=segment_source_str,
            roads_by_source=roads_by_source,
            segments_by_source=segments_by_source,
        )

        # ---------------- Segment-level outputs (imu, depth, gps, meta) ----------------
        self.write_segment_outputs(
            result=result, segment_target_dir=segment_target_dir)

        # ---------------- Segment media ----------------
        self.copy_segment_media(
            result=result,
            segment_source_dir=segment_source_dir,
            segment_target_dir=segment_target_dir,
        )

        # Pothole processing mutates the in-memory instance of manifest as a side effect,
        # so we persist the updated manifest at the end.
        # ---------------- Potholes ----------------
        self.process_potholes(
            segment_entry=segment_entry,
            segment_source_dir=segment_source_dir,
            segment_target_dir=segment_target_dir,
        )

        # Persist manifest
        self._write_json(manifest_path, manifest, indent=2)

        logger.info(
            "Exported segment to data2. source=%s target=%s",
            segment_source_dir,
            segment_target_dir,
        )

    def write_segment_outputs(self, *, result: SegmentProcessingResult, segment_target_dir: Path) -> None:
        """Write all segment-level JSON outputs into the target segment dir."""
        self.write_segment_imu(
            result=result, segment_target_dir=segment_target_dir)
        self.write_segment_depth_timestamps(
            result=result, segment_target_dir=segment_target_dir)
        self.write_segment_gps(
            result=result, segment_target_dir=segment_target_dir)
        self.write_segment_meta(
            result=result, segment_target_dir=segment_target_dir)

    def write_segment_imu(self, *, result: SegmentProcessingResult, segment_target_dir: Path) -> None:
        if result.imu_series is None:
            return
        self._write_json(
            segment_target_dir / "imu.json",
            [
                {
                    "vertical_displacement": e.vertical_displacement,
                    "video_timestamp": e.video_timestamp,
                }
                for e in result.imu_series.entries
            ],
            indent=2,
        )

    def write_segment_depth_timestamps(self, *, result: SegmentProcessingResult, segment_target_dir: Path) -> None:
        if result.depth_timeline is None:
            return
        self._write_json(
            segment_target_dir / "segment_depth_timestamps.json",
            [
                {
                    "pothole_depth": e.pothole_depth,
                    "video_timestamp": e.video_timestamp,
                }
                for e in result.depth_timeline.entries
            ],
            indent=4,
        )

    def write_segment_gps(self, *, result: SegmentProcessingResult, segment_target_dir: Path) -> None:
        if result.gps_series is None:
            return
        self._write_json(
            segment_target_dir / "segment_gps.json",
            [
                {
                    "timestamp": e.timestamp,
                    "lat": e.lat,
                    "lng": e.lng,
                    "alt": e.alt,
                }
                for e in result.gps_series.entries
            ],
            indent=2,
        )

    def write_segment_meta(self, *, result: SegmentProcessingResult, segment_target_dir: Path) -> None:
        """Write segment_meta.json (start/end/length)."""
        start_loc = result.gps_series.start_location if result.gps_series else ""
        end_loc = result.gps_series.end_location if result.gps_series else ""
        length_km = result.route_length.kilometers if result.route_length else 0.0
        self._write_json(
            segment_target_dir / "segment_meta.json",
            {
                "start_loc": start_loc,
                "end_loc": end_loc,
                "length_in_km": float(length_km),
            },
            indent=2,
        )

    def copy_segment_media(
        self,
        *,
        result: SegmentProcessingResult,
        segment_source_dir: Path,
        segment_target_dir: Path,
    ) -> None:
        """Copy segment-level media into the legacy target layout under Data/."""
        self.copy_segment_survey_video(
            result=result, segment_target_dir=segment_target_dir)
        self.copy_segment_lidar_scan(
            segment_source_dir=segment_source_dir,
            segment_target_dir=segment_target_dir,
        )

    def copy_segment_survey_video(self, *, result: SegmentProcessingResult, segment_target_dir: Path) -> None:
        """Copy the survey video into Data/Survey video/ (if present)."""
        if result.video is None or not result.video.path.is_file():
            return
        self._copy_file(
            result.video.path,
            segment_target_dir / "Data" / "Survey video" / result.video.path.name,
        )

    def copy_segment_lidar_scan(self, *, segment_source_dir: Path, segment_target_dir: Path) -> None:
        """Copy the first immediate .pcd found in the source segment dir into Data/Lidar Scan/."""
        lidar_scan = self._find_first_immediate_with_suffixes(
            segment_source_dir, (".pcd",))
        if lidar_scan is None or not lidar_scan.is_file():
            return
        self._copy_file(
            lidar_scan,
            segment_target_dir / "Data" / "Lidar Scan" / lidar_scan.name,
        )

    def process_potholes(
        self,
        *,
        segment_entry: Dict[str, Any],
        segment_source_dir: Path,
        segment_target_dir: Path,
    ) -> None:
        """Export potholes into `Potholes/pt-*` and update the segment manifest entry."""
        (segment_target_dir / "Potholes").mkdir(parents=True, exist_ok=True)

        pothole_dirs = self._discover_pothole_dirs(segment_source_dir)
        potholes_by_source = self.potholes_by_source(
            segment_entry=segment_entry)

        for pothole_dir in pothole_dirs:
            self.process_pothole(
                pothole_dir=pothole_dir,
                potholes_by_source=potholes_by_source,
                segment_target_dir=segment_target_dir,
            )

    def process_pothole(
        self,
        *,
        pothole_dir: Path,
        potholes_by_source: Dict[str, Dict[str, Any]],
        segment_target_dir: Path,
    ) -> None:
        """Export a single pothole dir into the legacy target layout."""
        pothole_entry = self.get_or_create_pothole_entry(
            potholes_by_source=potholes_by_source,
            pothole_dir=pothole_dir,
            segment_target_dir=segment_target_dir,
        )
        pothole_target_dir = Path(pothole_entry["target"])
        pothole_target_dir.mkdir(parents=True, exist_ok=True)

        self.copy_pothole_media(
            pothole_dir=pothole_dir,
            pothole_target_dir=pothole_target_dir,
            pothole_entry=pothole_entry,
        )
        self.write_pothole_meta(
            pothole_dir=pothole_dir,
            pothole_target_dir=pothole_target_dir,
        )

    def potholes_by_source(self, *, segment_entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Return the potholes map from the segment entry, normalizing to a dict."""
        potholes = segment_entry.get("potholes")
        if isinstance(potholes, dict):
            return potholes  # type: ignore[return-value]
        segment_entry["potholes"] = {}
        return segment_entry["potholes"]  # type: ignore[return-value]

    def pothole_source_key(self, *, pothole_dir: Path) -> str:
        """Stable key for manifest maps (absolute source dir path)."""
        return str(pothole_dir.resolve())

    def get_pothole_entry_from_dir(
        self,
        *,
        potholes_by_source: Dict[str, Dict[str, Any]],
        pothole_dir: Path,
    ) -> Dict[str, Any] | None:
        """Get an existing pothole entry for a given source pothole dir (if present)."""
        key = self.pothole_source_key(pothole_dir=pothole_dir)
        entry = potholes_by_source.get(key)
        return entry if isinstance(entry, dict) else None

    def get_or_create_pothole_entry(
        self,
        *,
        potholes_by_source: Dict[str, Dict[str, Any]],
        pothole_dir: Path,
        segment_target_dir: Path,
    ) -> Dict[str, Any]:
        """Retrieve the pothole entry from the manifest or create it if missing."""
        source_key = self.pothole_source_key(pothole_dir=pothole_dir)
        entry = self.get_pothole_entry_from_dir(
            potholes_by_source=potholes_by_source,
            pothole_dir=pothole_dir,
        )
        if entry is None:
            entry = self.create_pothole_entry(
                pothole_source_str=source_key,
                segment_target_dir=segment_target_dir,
            )
            potholes_by_source[source_key] = entry
            return entry

        # normalize minimal fields (defensive)
        entry.setdefault("source", source_key)
        entry.setdefault("image", "")
        entry.setdefault("pcd", "")
        if not isinstance(entry.get("id"), str) or not entry.get("id"):
            entry["id"] = _generate_prefixed_uuid("pt")
        if not isinstance(entry.get("target"), str) or not entry.get("target"):
            entry["target"] = str(
                (segment_target_dir / "Potholes" / entry["id"]).resolve())
        return entry

    def create_pothole_entry(
        self,
        *,
        pothole_source_str: str,
        segment_target_dir: Path,
    ) -> Dict[str, Any]:
        """Create a new pothole manifest entry."""
        pothole_id = _generate_prefixed_uuid("pt")
        target = str((segment_target_dir / "Potholes" / pothole_id).resolve())
        return {
            "id": pothole_id,
            "source": pothole_source_str,
            "target": target,
            "image": "",
            "pcd": "",
        }

    def copy_pothole_media(
        self,
        *,
        pothole_dir: Path,
        pothole_target_dir: Path,
        pothole_entry: Dict[str, Any],
    ) -> None:
        """Copy pothole image + pcd into the legacy target layout and update entry filenames."""
        img_src, pcd_src = self._find_pothole_media(pothole_dir)

        if img_src is not None and img_src.is_file():
            img_dst = pothole_target_dir / "Image" / img_src.name
            self._copy_file(img_src, img_dst)
            pothole_entry["image"] = img_src.name

        if pcd_src is not None and pcd_src.is_file():
            pcd_dst = pothole_target_dir / "Lidar Scan" / pcd_src.name
            self._copy_file(pcd_src, pcd_dst)
            pothole_entry["pcd"] = pcd_src.name

    def write_pothole_meta(self, *, pothole_dir: Path, pothole_target_dir: Path) -> None:
        """Write pothole_meta.json (from inspection_result.json)."""
        meta = self._extract_pothole_meta(pothole_dir)
        if meta is None:
            return
        self._write_json(pothole_target_dir /
                         "pothole_meta.json", meta, indent=2)

    def manifest_lookups(
        self,
        manifest: Dict[str, Any],
    ) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        """Return (roads_by_source, segments_by_source) from a v2 manifest."""
        roads_by_source: Dict[str, str] = manifest.setdefault("roads", {})
        segments_by_source: Dict[str, Dict[str, Any]
                                 ] = manifest.setdefault("segments", {})
        return roads_by_source, segments_by_source

    def ensure_output_locations(
        self,
        *,
        roads_root: Path,
        road_source_str: str,
        segment_source_str: str,
        roads_by_source: Dict[str, str],
        segments_by_source: Dict[str, Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Path]:
        """Ensure road+segment mappings exist and return (segment_entry, segment_target_dir)."""
        road_id = self.ensure_road_exists(
            road_source_str=road_source_str,
            roads_by_source=roads_by_source,
            roads_root=roads_root,
        )
        segment_entry, segment_target_dir = self.ensure_segment_exists(
            segment_source_str=segment_source_str,
            segments_by_source=segments_by_source,
            roads_by_source=roads_by_source,
            road_source_str=road_source_str,
            roads_root=roads_root,
            road_id=road_id,
        )
        return segment_entry, segment_target_dir

    def ensure_road_exists(
        self,
        road_source_str: str,
        roads_by_source: Dict[str, str],
        *,
        roads_root: Path,
    ) -> str:
        """Ensure road id exists and the target road folder exists. Returns road_id."""
        road_id = roads_by_source.get(road_source_str)
        if not road_id:
            road_id = _generate_prefixed_uuid("rd")
            roads_by_source[road_source_str] = road_id
        (roads_root / road_id).mkdir(parents=True, exist_ok=True)
        return road_id

    def ensure_segment_exists(
        self,
        segment_source_str: str,
        segments_by_source: Dict[str, Dict[str, Any]],
        roads_by_source: Dict[str, str],
        road_source_str: str,
        *,
        roads_root: Path,
        road_id: str,
    ) -> tuple[Dict[str, Any], Path]:
        """Ensure segment entry exists and its target folder exists."""
        segment_entry = segments_by_source.get(segment_source_str)
        if not isinstance(segment_entry, dict):
            segment_id = _generate_prefixed_uuid("rs")
            segment_target_dir = (roads_root / road_id / segment_id).resolve()
            segment_entry = {
                "id": segment_id,
                "source": segment_source_str,
                "road_source": road_source_str,
                "road_id": road_id,
                "target": str(segment_target_dir),
                "potholes": {},  # pothole_source_dir -> pothole entry
            }
            segments_by_source[segment_source_str] = segment_entry
        else:
            # keep entry consistent with current road mapping
            segment_entry.setdefault("source", segment_source_str)
            segment_entry.setdefault("road_source", road_source_str)
            segment_entry.setdefault("road_id", road_id)
            segment_entry.setdefault("potholes", {})

            seg_id = segment_entry.get("id")
            if not isinstance(seg_id, str) or not seg_id:
                seg_id = _generate_prefixed_uuid("rs")
                segment_entry["id"] = seg_id
            if "target" not in segment_entry:
                segment_entry["target"] = str(
                    (roads_root / road_id / seg_id).resolve())

        segment_target_dir = Path(segment_entry["target"])
        segment_target_dir.mkdir(parents=True, exist_ok=True)
        return segment_entry, segment_target_dir

    def _load_or_init_manifest_v2(
        self,
        *,
        manifest_path: Path,
        roads_root_guess: Path,
        target_roads_dir: Path,
    ) -> Dict[str, Any]:
        """Load manifest and normalize to a v2, O(1)-lookup structure.

        v2 schema:
        {
          "version": 2,
          "roads_root": "<source roads root guess>",
          "target_roads_dir": "<export_root>/Roads",
          "roads": { "<road_source_dir>": "rd-..." },
          "segments": {
            "<segment_source_dir>": {
              "id": "rs-...",
              "source": "<segment_source_dir>",
              "road_source": "<road_source_dir>",
              "road_id": "rd-...",
              "target": "<target_roads_dir>/rd-.../rs-...",
              "potholes": {
                "<pothole_source_dir>": {"id": "pt-...", "target": "...", "image": "", "pcd": ""}
              }
            }
          }
        }

        """
        if manifest_path.is_file():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    isinstance(raw, dict)
                    and raw.get("version") == 2
                    and isinstance(raw.get("roads"), dict)
                    and isinstance(raw.get("segments"), dict)
                ):
                    raw.setdefault("roads_root", str(
                        roads_root_guess.resolve()))
                    raw.setdefault("target_roads_dir", str(
                        target_roads_dir.resolve()))
                    raw.setdefault("roads", {})
                    raw.setdefault("segments", {})
                    return raw
            except Exception:
                logger.warning(
                    "Failed to parse existing manifest: %s. Recreating.", manifest_path
                )

        return {
            "version": 2,
            "roads_root": str(roads_root_guess.resolve()),
            "target_roads_dir": str(target_roads_dir.resolve()),
            "roads": {},    # road_source_dir -> rd-...
            "segments": {},  # segment_source_dir -> entry
        }


# ---------------------------------------------------------------------------
# Consolidated JSON Builder (produces trial_output.json)
# ---------------------------------------------------------------------------


@dataclass
class Data2ConsolidatedJsonBuilder:
    """Build a consolidated roads JSON from a Data2 target tree.

    This replaces the manual `build_road_json.py` step by producing the same
    output structure: {"roads": [...]} with each road containing segments,
    and each segment containing potholes, gps_data, depth_data, etc.

    The output JSON is written to the specified path (e.g. `trial_output.json`).
    """

    def build(self, *, target_roads_dir: Path, output_path: Path) -> Path:
        """Build consolidated JSON and return path to the output file."""
        roads_payload = []
        for road_dir in self._discover_roads(target_roads_dir):
            road_payload = self._build_road_payload(road_dir, target_roads_dir)
            roads_payload.append(road_payload)

        if not roads_payload:
            logger.warning(
                "No road payloads produced. Check target tree: %s", target_roads_dir)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"roads": roads_payload}, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Wrote consolidated JSON: %s (roads=%d, segments=%d)",
            output_path,
            len(roads_payload),
            sum(len(r.get("road_segments", [])) for r in roads_payload),
        )
        return output_path

    # ------------------------- discovery helpers -------------------------

    @staticmethod
    def _discover_roads(target_roads_dir: Path) -> List[Path]:
        """Discover road directories (rd-*) under the target tree."""
        if not target_roads_dir.is_dir():
            return []
        return sorted(
            d for d in target_roads_dir.iterdir()
            if d.is_dir() and d.name.startswith("rd-")
        )

    @staticmethod
    def _discover_segments(road_dir: Path) -> List[Path]:
        """Discover segment directories (rs-*) under a road."""
        return sorted(
            d for d in road_dir.iterdir()
            if d.is_dir() and d.name.startswith("rs-") and (d / "Potholes").is_dir()
        )

    # ------------------------- road building -------------------------

    def _build_road_payload(
        self,
        road_dir: Path,
        target_roads_dir: Path,
    ) -> Dict[str, Any]:
        """Build a single road payload."""
        road_id = road_dir.name
        segments = [
            self._build_segment_payload(seg_dir, road_id, target_roads_dir)
            for seg_dir in self._discover_segments(road_dir)
        ]
        return {
            "id": road_id,
            "name": "",
            "location": "",
            "road_segments": segments,
        }

    # ------------------------- segment building -------------------------

    def _build_segment_payload(
        self,
        segment_dir: Path,
        road_id: str,
        target_roads_dir: Path,
    ) -> Dict[str, Any]:
        """Build a single segment payload."""
        segment_id = segment_dir.name

        # Read optional segment_meta.json
        start_loc = ""
        end_loc = ""
        length_in_km = 0.0
        meta_path = segment_dir / "segment_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                start_loc = meta.get("start_loc", "") or ""
                end_loc = meta.get("end_loc", "") or ""
                length_in_km = float(meta.get("length_in_km", 0.0))
            except Exception as e:
                logger.warning(
                    "Failed to read segment_meta.json in %s: %s", segment_dir, e)

        # Survey video (Data/Survey video/*.mp4)
        survey_rel = self._find_asset_relpath(
            segment_dir / "Data" / "Survey video",
            (".mp4", ".mov", ".mkv", ".avi"),
            target_roads_dir,
        )

        # Lidar scan (Data/Lidar Scan/*.pcd)
        lidar_rel = self._find_asset_relpath(
            segment_dir / "Data" / "Lidar Scan",
            (".pcd",),
            target_roads_dir,
        )

        # gps_data (imu.json)
        gps_data = self._load_json_list(segment_dir / "imu.json")

        # depth_data (segment_depth_timestamps.json)
        depth_data = self._load_json_list(
            segment_dir / "segment_depth_timestamps.json")

        # potholes
        potholes = self._build_potholes(
            segment_dir, road_id, segment_id, target_roads_dir)

        return {
            "id": segment_id,
            "road_id": road_id,
            "iri": 0.0,
            "location": "",
            "start_loc": start_loc,
            "end_loc": end_loc,
            "length_in_km": length_in_km,
            "survey_video": survey_rel,
            "lidar_scan": lidar_rel,
            "gps_data": gps_data,
            "depth_data": depth_data,
            "potholes": potholes,
        }

    # ------------------------- pothole building -------------------------

    def _build_potholes(
        self,
        segment_dir: Path,
        road_id: str,
        segment_id: str,
        target_roads_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Build potholes array for a segment."""
        potholes_root = segment_dir / "Potholes"
        if not potholes_root.is_dir():
            return []

        # Load segment GPS for lat/lng lookup
        segment_gps = self._load_json_list(segment_dir / "segment_gps.json")

        potholes = []
        for pothole_dir in sorted(potholes_root.iterdir()):
            if not pothole_dir.is_dir():
                continue
            pothole_id = pothole_dir.name

            # Image path (Image/*.jpg or *.png)
            image_rel = self._find_asset_relpath(
                pothole_dir / "Image",
                (".jpg", ".png"),
                target_roads_dir,
            )

            # Lidar path (Lidar Scan/*.pcd)
            lidar_rel = self._find_asset_relpath(
                pothole_dir / "Lidar Scan",
                (".pcd",),
                target_roads_dir,
            )

            # Depth/volume/area from pothole_meta.json
            depth = 0.0
            volume = 0.0
            area = 0.0
            meta_path = pothole_dir / "pothole_meta.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    depth = float(meta.get("depth", 0.0) or 0.0)
                    volume = float(meta.get("volume", 0.0) or 0.0)
                    area = float(meta.get("area", 0.0) or 0.0)
                except Exception as e:
                    logger.warning(
                        "Failed to read pothole_meta.json in %s: %s", pothole_dir, e)

            # Lat/lng from GPS nearest to image timestamp
            lat, lng = self._get_pothole_position(
                pothole_dir / "Image", segment_gps)

            potholes.append({
                "id": pothole_id,
                "road_id": road_id,
                "road_segment_id": segment_id,
                "lat": lat,
                "lng": lng,
                "depth": depth,
                "volume": volume,
                "area": area,
                "image": image_rel,
                "lidar_scan": lidar_rel,
            })

        return potholes

    # ------------------------- utility helpers -------------------------

    @staticmethod
    def _find_asset_relpath(
        folder: Path,
        suffixes: Tuple[str, ...],
        target_roads_dir: Path,
    ) -> str:
        """Find first file with given suffixes in folder and return relpath."""
        if not folder.is_dir():
            return ""
        suffix_set = {s.lower() for s in suffixes}
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in suffix_set
        )
        if not files:
            return ""
        try:
            return str(files[0].relative_to(target_roads_dir))
        except ValueError:
            return str(files[0])

    @staticmethod
    def _load_json_list(path: Path) -> List[Dict[str, Any]]:
        """Load a JSON file expected to be a list; return [] if absent or invalid."""
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Failed to load JSON list from %s: %s", path, e)
            return []

    @staticmethod
    def _get_pothole_position(
        image_folder: Path,
        segment_gps: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """Get lat/lng for a pothole by finding GPS entry nearest to image timestamp."""
        if not image_folder.is_dir() or not segment_gps:
            return 0.0, 0.0

        # Find image timestamp
        image_ts: Optional[float] = None
        for f in image_folder.iterdir():
            if f.suffix.lower() in (".jpg", ".png"):
                try:
                    image_ts = float(f.stem.split(
                        ".")[0] + "." + f.stem.split(".")[1] if "." in f.stem else f.stem)
                except Exception:
                    pass
                break

        if image_ts is None:
            return 0.0, 0.0

        # Find nearest GPS entry
        try:
            nearest = min(
                segment_gps,
                key=lambda g: abs(float(g.get("timestamp", 0.0)) - image_ts)
            )
            return float(nearest.get("lat", 0.0)), float(nearest.get("lng", 0.0))
        except Exception:
            return 0.0, 0.0
