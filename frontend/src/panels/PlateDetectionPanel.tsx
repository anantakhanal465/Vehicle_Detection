import { useState } from "react";

import { detectPlates } from "../api";
import { FileField } from "../components/FileField";
import { ImageWithBoxes, type OverlayBox } from "../components/ImageWithBoxes";
import { SubmitButton } from "../components/SubmitButton";
import { PLATE_COLOR, vehicleColor } from "../colors";
import type { PlateDetectionResponse } from "../types";

export function PlateDetectionPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PlateDetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFileChange(next: File | null) {
    setFile(next);
    setResult(null);
    setError(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return next ? URL.createObjectURL(next) : null;
    });
  }

  async function handleSubmit() {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const response = await detectPlates(file);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed");
    } finally {
      setLoading(false);
    }
  }

  const boxes: OverlayBox[] = [];

  if (result) {
    for (const vehicle of result.vehicles) {
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

    for (const plate of result.unmatched_plates) {
      boxes.push({
        ...plate.bounding_box,
        label: plate.text ?? "plate (unmatched)",
        color: PLATE_COLOR,
      });
    }
  }

  const matchedReadable =
    result?.vehicles.filter((v) => v.license_plate?.text) ?? [];
  const unmatchedReadable =
    result?.unmatched_plates.filter((p) => p.text) ?? [];
  const readablePlateCount =
    matchedReadable.length + unmatchedReadable.length;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end gap-4">
        <FileField
          accept="image/*"
          file={file}
          onChange={handleFileChange}
          label="Image"
        />
        <SubmitButton onClick={handleSubmit} disabled={!file} loading={loading}>
          Detect plates
        </SubmitButton>
      </div>

      <p className="text-sm text-neutral-500 dark:text-neutral-400">
        Reads Nepali plates in both Devanagari and embossed Latin script.
        Accuracy depends heavily on how many pixels the plate itself
        occupies in the source image — distant traffic-camera shots will
        often localize the plate correctly but fail to read the text.
      </p>

      {error && (
        <p className="rounded-lg bg-red-100 px-4 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      {previewUrl && (
        <div className="flex flex-col gap-4 lg:flex-row">
          <ImageWithBoxes src={previewUrl} boxes={boxes} />

          {result && (
            <div className="min-w-[240px] flex-1">
              <h3 className="mb-2 text-sm font-semibold text-neutral-700 dark:text-neutral-300">
                {readablePlateCount} plate
                {readablePlateCount === 1 ? "" : "s"} read of{" "}
                {result.vehicles.length} vehicle
                {result.vehicles.length === 1 ? "" : "s"}
                {result.unmatched_plates.length > 0 &&
                  ` (+${result.unmatched_plates.length} plate${
                    result.unmatched_plates.length === 1 ? "" : "s"
                  } with no matching vehicle)`}
              </h3>
              <ul className="flex flex-col gap-1.5">
                {result.vehicles.map((v, i) => (
                  <li
                    key={`vehicle-${i}`}
                    className="flex items-center gap-2 rounded-md border border-neutral-200 px-3 py-1.5 text-sm dark:border-neutral-800"
                  >
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: vehicleColor(v.vehicle_type) }}
                    />
                    <span className="capitalize">{v.vehicle_type}</span>
                    <span className="ml-auto font-mono text-neutral-700 dark:text-neutral-200">
                      {v.license_plate?.text ?? "—"}
                    </span>
                  </li>
                ))}
                {result.unmatched_plates.map((p, i) => (
                  <li
                    key={`unmatched-${i}`}
                    className="flex items-center gap-2 rounded-md border border-dashed border-neutral-300 px-3 py-1.5 text-sm dark:border-neutral-700"
                  >
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: PLATE_COLOR }}
                    />
                    <span className="text-neutral-500 dark:text-neutral-400">
                      Plate (no vehicle detected)
                    </span>
                    <span className="ml-auto font-mono text-neutral-700 dark:text-neutral-200">
                      {p.text ?? "unreadable"}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
