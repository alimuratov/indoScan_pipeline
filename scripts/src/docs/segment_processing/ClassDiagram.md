
@startuml

' segment_processing context - Class Diagram
' Matches implementation in scripts/src/segment_processing/

' Notes:
' - Segment is a domain entity (identity + source_dir).
' - Workspace encapsulates path resolution (where to read/write).
' - Ports are side-effectful: they write files and return artifact receipts.
' - Use case orchestrates ports, handling errors gracefully (partial success).

' ============================================================================
' APPLICATION LAYER
' ============================================================================

class ProcessSegmentUseCase <<Application Service>> {
  +run(segment_dir: Path, config: ProcessSegmentConfig): SegmentProcessingResult
}

class ProcessSegmentConfig <<Value Object>> {
  +fps: int = 10
  +imu_interval_seconds: float = 30.0
  +persist_results: bool = True
}

ProcessSegmentUseCase ..> SegmentWorkspaceFactory : uses
ProcessSegmentUseCase ..> VideoCreator : uses
ProcessSegmentUseCase ..> ImuProcessor : uses
ProcessSegmentUseCase ..> RouteLengthCalculator : uses
ProcessSegmentUseCase ..> DepthTimelineExtractor : uses
ProcessSegmentUseCase ..> GpsConverter : uses
ProcessSegmentUseCase ..> SegmentArtifactsRepository : uses (optional)

' ============================================================================
' PORTS (Protocol interfaces)
' ============================================================================

interface SegmentWorkspaceFactory <<Port>> {
  +build(segment_dir: Path): SegmentWorkspace
}

interface SegmentWorkspace <<Port>> {
  ' Source paths
  +segment_dir: Path
  +raw_images_dir: Path
  +imu_txt: Path
  +gps_txt: Path
  +pothole_dirs: List[Path]
  ' Output paths
  +output_video_path: Path
  +imu_json_path: Path
  +route_length_json_path: Path
  +depth_timeline_json_path: Path
  +gps_json_path: Path
  +segment_meta_json_path: Path
}

interface VideoCreator <<Port>> {
  +create_video(images_dir: Path, output_path: Path, fps: int): VideoArtifact
}

interface ImuProcessor <<Port>> {
  +create_series(imu_file_path: Path, output_path: Path, interval_seconds: float): VerticalDisplacementSeries
}

interface RouteLengthCalculator <<Port>> {
  +calculate(odometry_file_path: Path, output_path: Path): RouteLengthResult
}

interface DepthTimelineExtractor <<Port>> {
  +extract(segment_dir: Path, raw_images_dir: Path, output_path: Path): DepthTimeline
}

interface GpsConverter <<Port>> {
  +convert(gps_file_path: Path, output_path: Path): GpsSeries
}

interface SegmentArtifactsRepository <<Port>> {
  +save(result: SegmentProcessingResult): None
}

' ============================================================================
' DOMAIN MODEL
' ============================================================================

class Segment <<Entity>> {
  +id: SegmentId
  +source_dir: Path
  +name: str {property}
}

class SegmentId <<Value Object>> {
  +value: str
}

enum SegmentProcessingStatus <<Enum>> {
  OK
  PARTIAL
  FAILED
}

class SegmentProcessingResult <<Value Object>> {
  +segment: Segment
  +status: SegmentProcessingStatus
  +video: VideoArtifact?
  +imu_series: VerticalDisplacementSeries?
  +route_length: RouteLengthResult?
  +depth_timeline: DepthTimeline?
  +gps_series: GpsSeries?
  +segment_meta: SegmentMeta?
  +warnings: List[str]
  +errors: List[str]
}

class VideoArtifact <<Value Object>> {
  +path: Path
  +fps: int
  +frame_count: int
  +duration_seconds: float
}

class VerticalDisplacementSeries <<Value Object>> {
  +entries: List[VerticalDisplacementEntry]
  +path: Path?
}

class VerticalDisplacementEntry <<Value Object>> {
  +video_timestamp: str
  +vertical_displacement: float
}

class RouteLengthResult <<Value Object>> {
  +meters: float
  +kilometers: float
  +path: Path?
}

class DepthTimeline <<Value Object>> {
  +entries: List[DepthTimelineEntry]
  +path: Path?
}

class DepthTimelineEntry <<Value Object>> {
  +video_timestamp: str
  +pothole_depth: float
}

class GpsSeries <<Value Object>> {
  +entries: List[GpsEntry]
  +path: Path?
  +start_location: str {property}
  +end_location: str {property}
}

class GpsEntry <<Value Object>> {
  +timestamp: float
  +lat: float
  +lng: float
  +alt: float?
}

class SegmentMeta <<Value Object>> {
  +start_loc: str
  +end_loc: str
  +length_in_km: float
  +path: Path?
}

' ============================================================================
' INFRASTRUCTURE (Concrete implementations)
' ============================================================================

class FilesystemSegmentWorkspaceFactory <<Adapter>> {
  +raw_images_folder_name: str
  +imu_filename: str
  +gps_filename: str
  +pothole_prefix: str
  +output_dir: Path?
  +build(segment_dir: Path): FilesystemSegmentWorkspace
}

class FilesystemSegmentWorkspace <<Adapter>> {
  ' Implements SegmentWorkspace protocol
}

class OpenCVVideoCreator <<Adapter>> {
  ' Implements VideoCreator protocol
  ' Uses cv2 to create mp4 from images
}

class LegacyImuProcessor <<Adapter>> {
  ' Implements ImuProcessor protocol
  ' Wraps legacy sensors.process_gps logic
}

class LegacyRouteLengthCalculator <<Adapter>> {
  ' Implements RouteLengthCalculator protocol
  ' Wraps legacy sensors.calculate_road_length logic
}

class LegacyDepthTimelineExtractor <<Adapter>> {
  ' Implements DepthTimelineExtractor protocol
  ' Reads inspection_result.json
}

class FilesystemGpsConverter <<Adapter>> {
  ' Implements GpsConverter protocol
}

class FilesystemSegmentArtifactsRepository <<Adapter>> {
  ' Implements SegmentArtifactsRepository protocol
  ' Currently a no-op (ports persist themselves)
  ' Extensible for target tree export
}

class Data2SegmentArtifactsRepository <<Adapter>> {
  ' Implements SegmentArtifactsRepository protocol
  ' Exports legacy-compatible `output/data2` target tree:
  ' - Generates rd/rs/pt IDs (manifested)
  ' - Writes copy_manifest.json and Roads/rd-*/rs-* structure
}

' ============================================================================
' RELATIONSHIPS
' ============================================================================

Segment --> SegmentId
SegmentProcessingResult --> Segment
SegmentProcessingResult --> SegmentProcessingStatus
SegmentProcessingResult --> VideoArtifact
SegmentProcessingResult --> VerticalDisplacementSeries
SegmentProcessingResult --> RouteLengthResult
SegmentProcessingResult --> DepthTimeline
SegmentProcessingResult --> GpsSeries
SegmentProcessingResult --> SegmentMeta

VerticalDisplacementSeries --> VerticalDisplacementEntry
DepthTimeline --> DepthTimelineEntry
GpsSeries --> GpsEntry

FilesystemSegmentWorkspaceFactory ..|> SegmentWorkspaceFactory
FilesystemSegmentWorkspace ..|> SegmentWorkspace
OpenCVVideoCreator ..|> VideoCreator
LegacyImuProcessor ..|> ImuProcessor
LegacyRouteLengthCalculator ..|> RouteLengthCalculator
LegacyDepthTimelineExtractor ..|> DepthTimelineExtractor
FilesystemGpsConverter ..|> GpsConverter
FilesystemSegmentArtifactsRepository ..|> SegmentArtifactsRepository
Data2SegmentArtifactsRepository ..|> SegmentArtifactsRepository

@enduml
