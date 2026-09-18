import type {
  HistoryRecord,
  ImageDetectionResponse,
  PlateDetectionResponse,
  VehicleType,
  VideoDetectionResponse,
} from "./types";

async function parseOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail =
      (body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : null) ?? `Request failed (${response.status})`;
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export async function detectImage(
  file: File,
  vehicleType: VehicleType,
): Promise<ImageDetectionResponse> {
  const formData = new FormData();
  formData.append("vehicle_type", vehicleType);
  formData.append("file", file);

  const response = await fetch("/api/detection/image", {
    method: "POST",
    body: formData,
  });

  return parseOrThrow<ImageDetectionResponse>(response);
}

export async function detectPlates(
  file: File,
): Promise<PlateDetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/detection/plate", {
    method: "POST",
    body: formData,
  });

  return parseOrThrow<PlateDetectionResponse>(response);
}

export async function detectVideo(
  file: File,
  vehicleType: VehicleType,
  readPlates: boolean,
): Promise<VideoDetectionResponse> {
  const formData = new FormData();
  formData.append("vehicle_type", vehicleType);
  formData.append("read_plates", String(readPlates));
  formData.append("file", file);

  const response = await fetch("/api/detection/video", {
    method: "POST",
    body: formData,
  });

  return parseOrThrow<VideoDetectionResponse>(response);
}

export async function fetchHistory(limit = 20): Promise<HistoryRecord[]> {
  const response = await fetch(`/api/detection/history?limit=${limit}`);
  return parseOrThrow<HistoryRecord[]>(response);
}
 