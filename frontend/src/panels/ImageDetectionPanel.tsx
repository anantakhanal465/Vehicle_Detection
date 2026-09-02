import { useState } from "react";

import { detectImage } from "../api";
import { FileField } from "../components/FileField";
import { ImageWithBoxes } from "../components/ImageWithBoxes";
import { SubmitButton } from "../components/SubmitButton";
import { VehicleTypeSelect } from "../components/VehicleTypeSelect";
import { vehicleColor } from "../colors";
import type { ImageDetectionResponse, VehicleType } from "../types";

export function ImageDetectionPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [vehicleType, setVehicleType] = useState<VehicleType>("all");
  const [result, setResult] = useState<ImageDetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFileChange(next: File | null) {
    setFile(next);
    setResult(null);
    setError(null);
    setPreviewUrl(next ? URL.createObjectURL(next) : null);
  }

  async function handleSubmit() {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const response = await detectImage(file, vehicleType);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end gap-4">
        <FileField
          accept="image/*"
          file={file}
          onChange={handleFileChange}
          label="Image"
        />
        <VehicleTypeSelect value={vehicleType} onChange={setVehicleType} />
        <SubmitButton onClick={handleSubmit} disabled={!file} loading={loading}>
          Detect vehicles
        </SubmitButton>
      </div>

      {error && (
        <p className="rounded-lg bg-red-100 px-4 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      {previewUrl && (
        <div className="flex flex-col gap-4 lg:flex-row">
          <ImageWithBoxes
            src={previewUrl}
            boxes={(result?.detections ?? []).map((d) => ({
              ...d.bounding_box,
              label: `${d.vehicle_type} ${d.confidence.toFixed(2)}`,
              color: vehicleColor(d.vehicle_type),
            }))}
          />

          {result && (
            <div className="min-w-[220px] flex-1">
              <h3 className="mb-2 text-sm font-semibold text-neutral-700 dark:text-neutral-300">
                {result.detections.length} detection
                {result.detections.length === 1 ? "" : "s"}
              </h3>
              <ul className="flex flex-col gap-1.5">
                {result.detections.map((d, i) => (
                  <li
                    key={i}
                    className="flex items-center gap-2 rounded-md border border-neutral-200 px-3 py-1.5 text-sm dark:border-neutral-800"
                  >
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: vehicleColor(d.vehicle_type) }}
                    />
                    <span className="capitalize">{d.vehicle_type}</span>
                    <span className="ml-auto text-neutral-500 dark:text-neutral-400">
                      {(d.confidence * 100).toFixed(1)}%
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
