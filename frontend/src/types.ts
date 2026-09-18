export type VehicleType = "all" | "car" | "motorcycle" | "bus" | "truck";

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  vehicle_type: string;
  confidence: number;
  bounding_box: BoundingBox;
}

export interface ImageDetectionResponse {
  success: boolean;
  vehicle_type: string;
  detections: Detection[];
}

export interface PlateReading {
  text: string | null;
  detection_confidence: number;
  ocr_confidence: number | null;
  bounding_box: BoundingBox;
}

export interface VehicleWithPlate extends Detection {
  license_plate: PlateReading | null;
}

export interface PlateDetectionResponse {
  success: boolean;
  vehicles: VehicleWithPlate[];
  unmatched_plates: PlateReading[];
}

export interface TrackPlate {
  text: string;
  confidence: number;
}

export interface VideoDetectionResponse {
  success: boolean;
  vehicle_type: string;
  frames_processed: number;
  unique_vehicles: number;
  download_url: string;
  plates: Record<string, TrackPlate>;
}

export interface HistoryRecord {
  id: number;
  created_at: string;
  source_type: "image" | "video" | "plate";
  vehicle_type: string;
  detections: unknown;
  frames_processed: number | null;
  unique_vehicles: number | null;
  download_url: string | null;
  image_url: string | null;
}
 