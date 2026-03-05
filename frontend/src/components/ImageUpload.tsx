import { useRef, useState } from "react";

export type CropArea = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type ImageUploadProps = {
  previewUrl: string | null;
  selectedFile: File | null;
  loading: boolean;
  cropArea: CropArea | null;
  onFileChange: (file: File | null) => void;
  onCropAreaChange: (cropArea: CropArea | null) => void;
  onRunInference: () => void;
};

export default function ImageUpload({
  previewUrl,
  selectedFile,
  loading,
  cropArea,
  onFileChange,
  onCropAreaChange,
  onRunInference
}: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [draftCropArea, setDraftCropArea] = useState<CropArea | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const getFirstValidImage = (fileList: FileList | null): File | null => {
    if (!fileList || fileList.length === 0) {
      return null;
    }
    const droppedFile = fileList[0];
    return droppedFile.type.startsWith("image/") ? droppedFile : null;
  };

  const resetCrop = () => {
    setDrawStart(null);
    setDraftCropArea(null);
    onCropAreaChange(null);
  };

  const updateFile = (file: File | null) => {
    resetCrop();
    onFileChange(file);
  };

  const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

  const getNormalizedPointerPosition = (
    event: React.PointerEvent<HTMLDivElement>
  ): { x: number; y: number } | null => {
    const image = imageRef.current;
    if (!image) {
      return null;
    }

    const rect = image.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return null;
    }

    const x = clamp01((event.clientX - rect.left) / rect.width);
    const y = clamp01((event.clientY - rect.top) / rect.height);
    return { x, y };
  };

  const buildCropArea = (
    start: { x: number; y: number },
    end: { x: number; y: number }
  ): CropArea => {
    const left = Math.min(start.x, end.x);
    const top = Math.min(start.y, end.y);
    const right = Math.max(start.x, end.x);
    const bottom = Math.max(start.y, end.y);
    return {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top
    };
  };

  const activeCropArea = draftCropArea ?? cropArea;

  return (
    <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div>
        <label
          htmlFor="image-input"
          className="mb-2 block text-sm font-medium text-slate-700"
        >
          Upload image
        </label>
        <input
          ref={fileInputRef}
          id="image-input"
          type="file"
          accept="image/*"
          onChange={(event) => updateFile(getFirstValidImage(event.target.files))}
          className="block w-full text-sm text-slate-700 file:mr-3 file:rounded-md file:border-0 file:bg-slate-800 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-slate-700"
        />
      </div>

      <div
        role={!previewUrl ? "button" : undefined}
        tabIndex={!previewUrl ? 0 : undefined}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
          setIsDragging(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          const file = getFirstValidImage(event.dataTransfer.files);
          updateFile(file);
        }}
        onClick={() => {
          if (!previewUrl) {
            fileInputRef.current?.click();
          }
        }}
        onKeyDown={(event) => {
          if (!previewUrl && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            fileInputRef.current?.click();
          }
        }}
        className={`flex min-h-52 cursor-pointer items-center justify-center overflow-hidden rounded-lg border border-dashed p-3 transition ${
          isDragging
            ? "border-blue-500 bg-blue-50"
            : "border-slate-300 bg-slate-50 hover:bg-slate-100"
        }`}
      >
        {previewUrl ? (
          <div
            className="relative inline-block max-w-full select-none cursor-crosshair"
            onPointerDown={(event) => {
              if (loading || event.button !== 0) {
                return;
              }
              event.currentTarget.setPointerCapture(event.pointerId);
              const position = getNormalizedPointerPosition(event);
              if (!position) {
                return;
              }
              setDrawStart(position);
              setDraftCropArea(null);
            }}
            onPointerMove={(event) => {
              if (!drawStart || loading) {
                return;
              }
              const position = getNormalizedPointerPosition(event);
              if (!position) {
                return;
              }
              setDraftCropArea(buildCropArea(drawStart, position));
            }}
            onPointerUp={(event) => {
              if (!drawStart || loading) {
                return;
              }
              event.currentTarget.releasePointerCapture(event.pointerId);
              const position = getNormalizedPointerPosition(event);
              const nextCrop = position ? buildCropArea(drawStart, position) : null;
              setDrawStart(null);
              setDraftCropArea(null);

              if (!nextCrop || nextCrop.width < 0.02 || nextCrop.height < 0.02) {
                onCropAreaChange(null);
                return;
              }
              onCropAreaChange(nextCrop);
            }}
            onPointerCancel={() => {
              if (drawStart) {
                setDrawStart(null);
                setDraftCropArea(null);
              }
            }}
          >
            <img
              ref={imageRef}
              src={previewUrl}
              alt="Selected preview"
              draggable={false}
              onDragStart={(event) => event.preventDefault()}
              className="max-h-72 w-auto rounded-md object-contain"
            />
            {activeCropArea && (
              <div
                className="pointer-events-none absolute border-2 border-blue-500 bg-blue-500/15"
                style={{
                  left: `${activeCropArea.x * 100}%`,
                  top: `${activeCropArea.y * 100}%`,
                  width: `${activeCropArea.width * 100}%`,
                  height: `${activeCropArea.height * 100}%`
                }}
              />
            )}
          </div>
        ) : (
          <p className="text-sm text-slate-500">
            Drag and drop an image here, or click to choose a file.
          </p>
        )}
      </div>

      {previewUrl && (
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs text-slate-500">
            Drag on the image to draw crop area. Inference uses crop if selected.
          </p>
          <button
            type="button"
            onClick={resetCrop}
            disabled={!cropArea || loading}
            className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Clear crop
          </button>
        </div>
      )}

      <button
        type="button"
        disabled={!selectedFile || loading}
        onClick={onRunInference}
        className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-400"
      >
        {loading ? "Running inference..." : "Run inference"}
      </button>
    </section>
  );
}
