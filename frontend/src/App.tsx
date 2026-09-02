import { useState } from "react";

import { HistoryPanel } from "./panels/HistoryPanel";
import { ImageDetectionPanel } from "./panels/ImageDetectionPanel";
import { PlateDetectionPanel } from "./panels/PlateDetectionPanel";
import { VideoDetectionPanel } from "./panels/VideoDetectionPanel";

const TABS = [
  { id: "image", label: "Image", panel: ImageDetectionPanel },
  { id: "plate", label: "Plates", panel: PlateDetectionPanel },
  { id: "video", label: "Video", panel: VideoDetectionPanel },
  { id: "history", label: "History", panel: HistoryPanel },
] as const;

type TabId = (typeof TABS)[number]["id"];

function App() {
  const [activeTab, setActiveTab] = useState<TabId>("image");
  const ActivePanel = TABS.find((t) => t.id === activeTab)!.panel;

  return (
    <div className="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <header className="border-b border-neutral-200 dark:border-neutral-800">
        <div className="mx-auto max-w-5xl px-6 py-6">
          <h1 className="text-xl font-semibold">Vehicle Detection — Nepal</h1>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            YOLO vehicle detection, tracking, and license plate recognition
            tuned for Nepali plates.
          </p>
        </div>
      </header>

      <nav className="border-b border-neutral-200 dark:border-neutral-800">
        <div className="mx-auto flex max-w-5xl gap-1 px-6">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`border-b-2 px-4 py-3 text-sm font-medium transition ${
                activeTab === tab.id
                  ? "border-emerald-600 text-emerald-600 dark:border-emerald-400 dark:text-emerald-400"
                  : "border-transparent text-neutral-500 hover:text-neutral-800 dark:text-neutral-400 dark:hover:text-neutral-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="mx-auto max-w-5xl px-6 py-8">
        <ActivePanel />
      </main>
    </div>
  );
}

export default App;
