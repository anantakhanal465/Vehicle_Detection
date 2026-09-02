import { useEffect, useRef, useState } from "react";

import { detectVideo } from "../api";
import { FileField } from "../components/FileField";
import { SubmitButton } from "../components/SubmitButton";
import { VehicleTypeSelect } from "../components/VehicleTypeSelect";
import type { VehicleType, VideoDetectionResponse } from "../types";

export function VideoDetectionPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [vehicleType, setVehicleType] = useState<VehicleType>("all");
  const [readPlates, setReadPlates] = useState(false);
  const [result, setResult] = useState<VideoDetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!loading) {
      if (timerRef.current) window.clearInterval(timerRef.current);
      return;
    }

    setElapsed(0);
    const start = Date.now();
    timerRef.current = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);

    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, [loading]);

  function handleFileChange(next: File | null) {
    setFile(next);
    setResult(null);
    setError(null);
  }

  async function handleSubmit() {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await detectVideo(file, vehicleType, readPlates);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed");
    } finally {
      setLoading(false);
    }
  }

  const plateEntries = result ? Object.entries(result.plates) : [];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end gap-4">
        <FileField
          accept="video/*"
          file={file}
          onChange={handleFileChange}
          label="Video"
        />
        <VehicleTypeSelect value={vehicleType} onChange={setVehicleType} />
        <label className="flex items-center gap-2 pb-2.5 text-sm text-neutral-700 dark:text-neutral-300">
          <input
            type="checkbox"
            checked={readPlates}
            onChange={(event) => setReadPlates(event.target.checked)}
            className="h-4 w-4 accent-emerald-600"
          />
          Read license plates
        </label>
        <SubmitButton
          onClick={handleSubmit}
          disabled={!file}
          loading={loading}
          loadingText={`Processing… ${elapsed}s`}
        >
          Process video
        </SubmitButton>
      </div>

      <p className="text-sm text-neutral-500 dark:text-neutral-400">
        Detection + tracking alone takes roughly as long as the video's
        duration on CPU. With plate reading enabled it takes noticeably
        longer, since each tracked vehicle is retried periodically until a
        confident reading is found or a retry cap is hit — expect a minute
        or more even for short clips.
      </p>

      {error && (
        <p className="rounded-lg bg-red-100 px-4 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      {result && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-4 text-sm">
            <Stat label="Frames processed" value={result.frames_processed} />
            <Stat label="Unique vehicles" value={result.unique_vehicles} />
            <Stat label="Plates read" value={plateEntries.length} />
          </div>

          <video
            src={result.download_url}
            controls
            className="max-w-full rounded-lg border border-neutral-300 dark:border-neutral-700"
          />

          {plateEntries.length > 0 && (
            <div>
              <h3 className="mb-2 text-sm font-semibold text-neutral-700 dark:text-neutral-300">
                Plate readings by track
              </h3>
              <ul className="flex flex-col gap-1.5">
                {plateEntries.map(([trackId, plate]) => (
                  <li
                    key={trackId}
                    className="flex items-center gap-2 rounded-md border border-neutral-200 px-3 py-1.5 text-sm dark:border-neutral-800"
                  >
                    <span className="text-neutral-500 dark:text-neutral-400">
                      ID {trackId}
                    </span>
                    <span className="ml-auto font-mono text-neutral-700 dark:text-neutral-200">
                      {plate.text}
                    </span>
                    <span className="text-neutral-500 dark:text-neutral-400">
                      {(plate.confidence * 100).toFixed(1)}%
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

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-neutral-200 px-4 py-2 dark:border-neutral-800">
      <div className="text-xs text-neutral-500 dark:text-neutral-400">
        {label}
      </div>
      <div className="text-lg font-semibold text-neutral-800 dark:text-neutral-100">
        {value}
      </div>
    </div>
  );
}
