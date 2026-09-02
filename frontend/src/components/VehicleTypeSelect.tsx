import type { VehicleType } from "../types";

const OPTIONS: VehicleType[] = ["all", "car", "motorcycle", "bus", "truck"];

interface VehicleTypeSelectProps {
  value: VehicleType;
  onChange: (value: VehicleType) => void;
}

export function VehicleTypeSelect({ value, onChange }: VehicleTypeSelectProps) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
        Vehicle type
      </span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as VehicleType)}
        className="rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm text-neutral-800 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-200"
      >
        {OPTIONS.map((option) => (
          <option key={option} value={option}>
            {option === "all" ? "All vehicles" : option}
          </option>
        ))}
      </select>
    </label>
  );
}
