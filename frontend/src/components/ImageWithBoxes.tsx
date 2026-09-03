import { useState } from "react";

export interface OverlayBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  color: string;
}

interface ImageWithBoxesProps {
  src: string;
  boxes: OverlayBox[];
}

export function ImageWithBoxes({ src, boxes }: ImageWithBoxesProps) {
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(
    null,
  );

  return (
    <div className="relative inline-block max-w-full self-start leading-none">
      <img
        src={src}
        alt="Detection result"
        className="max-w-full rounded-lg border border-neutral-300 dark:border-neutral-700"
        onLoad={(event) => {
          const img = event.currentTarget;
          setNatural({ w: img.naturalWidth, h: img.naturalHeight });
        }}
      />
      {natural && (
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${natural.w} ${natural.h}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {boxes.map((box, index) => {
            const width = box.x2 - box.x1;
            const height = box.y2 - box.y1;
            const fontSize = Math.max(natural.h / 45, 14);
            const strokeWidth = Math.max(natural.w / 400, 2);

            return (
              <g key={index}>
                <rect
                  x={box.x1}
                  y={box.y1}
                  width={width}
                  height={height}
                  fill="none"
                  stroke={box.color}
                  strokeWidth={strokeWidth}
                />
                <text
                  x={box.x1}
                  y={Math.max(box.y1 - fontSize * 0.4, fontSize)}
                  fill={box.color}
                  fontSize={fontSize}
                  fontWeight={700}
                  paintOrder="stroke"
                  stroke="black"
                  strokeWidth={fontSize * 0.18}
                >
                  {box.label}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
