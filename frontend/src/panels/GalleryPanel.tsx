import { useEffect, useState } from "react";

import { fetchHistory } from "../api";
import { ImageWithBoxes, type OverlayBox } from "../components/ImageWithBoxes";
import { PLATE_COLOR, vehicleColor } from "../colors";
import type { Detection, HistoryRecord, VehicleWithPlate } from "../types";

const SOURCE_LABELS: Record<"image" | "plate", string> = {
  image: "Image",
  plate: "Plates",
};

export function GalleryPanel() {
  const [records, setRecords] = useState<HistoryRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);

    try {
      setRecords(await fetchHistory(100));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load gallery");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  const withImages = (records ?? []).filter(
    (r): r is HistoryRecord & { image_url: string } =>
      r.image_url != null && (r.source_type === "image" || r.source_type === "plate"),
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          Annotated results from past image and plate detections, newest
          first.
        </p>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          className="rounded-lg border border-neutral-300 px-3 py-1.5 text-sm font-medium text-neutral-700 hover:bg-neutral-100 disabled:opacity-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-900"
        >
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && (
        <p className="rounded-lg bg-red-100 px-4 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      {records && withImages.length === 0 && (
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          No results yet — try the Image or Plates tabs.
        </p>
      )}

      {withImages.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {withImages.map((record) => (
            <GalleryCard key={record.id} record={record} />
          ))}
        </div>
      )}
    </div>
  );
}

function GalleryCard({
  record,
}: {
  record: HistoryRecord & { image_url: string };
}) {
  const boxes = buildBoxes(record);

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-neutral-200 p-3 dark:border-neutral-800">
      <ImageWithBoxes src={record.image_url} boxes={boxes} />
      <div className="flex items-center justify-between text-xs text-neutral-500 dark:text-neutral-400">
        <span>{SOURCE_LABELS[record.source_type as "image" | "plate"]}</span>
        <span>{new Date(record.created_at).toLocaleString()}</span>
      </div>
    </div>
  );
}

function buildBoxes(record: HistoryRecord): OverlayBox[] {
  const boxes: OverlayBox[] = [];

  if (record.source_type === "image") {
    for (const d of asDetections(record.detections)) {
      boxes.push({
        ...d.bounding_box,
        label: `${d.vehicle_type} ${d.confidence.toFixed(2)}`,
        color: vehicleColor(d.vehicle_type),
      });
    }
  }

  if (record.source_type === "plate") {
    for (const vehicle of asVehiclesWithPlate(record.detections)) {
      boxes.push({
        ...vehicle.bounding_box,
        label: `${vehicle.vehicle_type} ${vehicle.confidence.toFixed(2)}`,
        color: vehicleColor(vehicle.vehicle_type),
      });

      if (vehicle.license_plate) {
        boxes.push({
          ...vehicle.license_plate.bounding_box,
          label: vehicle.license_plate.text ?? "plate (unreadable)",
          color: PLATE_COLOR,
        });
      }
    }
  }

  return boxes;
}

function asDetections(value: unknown): Detection[] {
  return Array.isArray(value) ? (value as Detection[]) : [];
}

function asVehiclesWithPlate(value: unknown): VehicleWithPlate[] {
  return Array.isArray(value) ? (value as VehicleWithPlate[]) : [];
}
 