interface FileFieldProps {
  accept: string;
  file: File | null;
  onChange: (file: File | null) => void;
  label: string;
}

export function FileField({ accept, file, onChange, label }: FileFieldProps) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
        {label}
      </span>
      <input
        type="file"
        accept={accept}
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
        className="block w-full cursor-pointer rounded-lg border border-neutral-300 bg-white text-sm text-neutral-700 file:mr-4 file:cursor-pointer file:rounded-md file:border-0 file:bg-emerald-600 file:px-4 file:py-2 file:font-medium file:text-white hover:file:bg-emerald-500 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-300"
      />
      {file && (
        <span className="text-xs text-neutral-500 dark:text-neutral-400">
          {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)
        </span>
      )}
    </label>
  );
}
