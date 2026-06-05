import { cn } from "@/lib/utils";

// TechSpar 双峰山形标(矢量,无白边)。炭峰跟随 currentColor(浅色主题→近黑、深色背景→近白),
// 青峰用 --logo-teal:浅色保留原图深青、深色换亮青,避免在近黑背景上糊掉。
export default function Logo({ className }) {
  return (
    <svg
      viewBox="0 0 838 754"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="TechSpar"
      shapeRendering="geometricPrecision"
      className={cn("shrink-0 block", className)}
    >
      <polygon
        fill="currentColor"
        points="756,252 505,452 348,266 211,375 317,376 491,582 756,371"
      />
      <polygon
        fill="var(--logo-teal)"
        points="647,279 538,278 366,74 102,284 102,398 351,202 510,386"
      />
    </svg>
  );
}
