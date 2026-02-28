type ImageUploadProps = {
  previewUrl: string | null;
  selectedFile: File | null;
  loading: boolean;
  onFileChange: (file: File | null) => void;
  onRunInference: () => void;
};

export default function ImageUpload({
  previewUrl,
  selectedFile,
  loading,
  onFileChange,
  onRunInference
}: ImageUploadProps) {
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
          id="image-input"
          type="file"
          accept="image/*"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
          className="block w-full text-sm text-slate-700 file:mr-3 file:rounded-md file:border-0 file:bg-slate-800 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-slate-700"
        />
      </div>

      <div className="flex min-h-52 items-center justify-center overflow-hidden rounded-lg border border-dashed border-slate-300 bg-slate-50 p-3">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Selected preview"
            className="max-h-48 w-auto rounded-md object-contain"
          />
        ) : (
          <p className="text-sm text-slate-500">Select an image to see preview.</p>
        )}
      </div>

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
