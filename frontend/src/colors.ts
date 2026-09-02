const VEHICLE_COLORS: Record<string, string> = {
  car: "#22c55e",
  motorcycle: "#38bdf8",
  bus: "#f59e0b",
  truck: "#f472b6",
};

export function vehicleColor(vehicleType: string): string {
  return VEHICLE_COLORS[vehicleType] ?? "#a3a3a3";
}

export const PLATE_COLOR = "#facc15";
