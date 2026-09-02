import { useEffect, useState } from "react";

import { fetchHistory } from "../api";
import type { HistoryRecord } from "../types";

const SOURCE_LABELS: Record<HistoryRecord["source_type"], string> = {
  image: "Image",
  video: "Video",
  plate: "Plate",
};

export function HistoryPanel() {
  const [records, setRecords] = useState<HistoryRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  async function load() {
    setLoading(true);
    setError(null);

    try {
      setRecords(await fetchHistory(50));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          Most recent 50 detection runs, newest first.
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

      {records && records.length === 0 && (
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          No detections yet — try the Image, Plates, or Video tabs.
        </p>
      )}

      {records && records.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-neutral-200 dark:border-neutral-800">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-neutral-200 bg-neutral-50 text-neutral-600 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-400">
              <tr>
                <th className="px-4 py-2 font-medium">Time</th>
                <th className="px-4 py-2 font-medium">Type</th>
                <th className="px-4 py-2 font-medium">Filter</th>
                <th className="px-4 py-2 font-medium">Summary</th>
                <th className="px-4 py-2 font-medium">Video</th>
              </tr>
            </thead>
            <tbody>
              {records.map((record) => (
                <RecordRow
                  key={record.id}
                  record={record}
                  expanded={expandedId === record.id}
                  onToggle={() =>
                    setExpandedId(expandedId === record.id ? null : record.id)
                  }
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function RecordRow({
  record,
  expanded,
  onToggle,
}: {
  record: HistoryRecord;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        className="cursor-pointer border-b border-neutral-100 last:border-0 hover:bg-neutral-50 dark:border-neutral-900 dark:hover:bg-neutral-900"
      >
        <td className="px-4 py-2 whitespace-nowrap text-neutral-500 dark:text-neutral-400">
          {new Date(record.created_at).toLocaleString()}
        </td>
        <td className="px-4 py-2">{SOURCE_LABELS[record.source_type]}</td>
        <td className="px-4 py-2 capitalize">{record.vehicle_type}</td>
        <td className="px-4 py-2 text-neutral-600 dark:text-neutral-300">
          {summarize(record)}
        </td>
        <td className="px-4 py-2">
          {record.download_url && (
            <a
              href={record.download_url}
              download
              onClick={(e) => e.stopPropagation()}
              className="text-emerald-600 hover:underline dark:text-emerald-400"
            >
              Download
            </a>
          )}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-neutral-100 bg-neutral-50 dark:border-neutral-900 dark:bg-neutral-900/50">
          <td colSpan={5} className="px-4 py-3">
            <pre className="max-h-64 overflow-auto text-xs text-neutral-600 dark:text-neutral-300">
              {JSON.stringify(record.detections, null, 2)}
            </pre>
          </td>
        </tr>
      )}
    </>
  );
}

function summarize(record: HistoryRecord): string {
  if (record.source_type === "video") {
    const plateCount =
      record.detections && typeof record.detections === "object"
        ? Object.keys(record.detections as object).length
        : 0;

    return `${record.frames_processed ?? 0} frames, ${
      record.unique_vehicles ?? 0
    } vehicles${plateCount ? `, ${plateCount} plates read` : ""}`;
  }

  if (Array.isArray(record.detections)) {
    if (record.source_type === "plate") {
      const withPlate = record.detections.filter(
        (v) =>
          v &&
          typeof v === "object" &&
          "license_plate" in v &&
          (v as { license_plate: { text: string | null } | null })
            .license_plate?.text,
      ).length;
      return `${withPlate} of ${record.detections.length} plates read`;
    }

    return `${record.detections.length} detection${
      record.detections.length === 1 ? "" : "s"
    }`;
  }

  return "—";
}
