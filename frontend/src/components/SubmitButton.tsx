interface SubmitButtonProps {
  onClick: () => void;
  disabled: boolean;
  loading: boolean;
  loadingText?: string;
  children: React.ReactNode;
}

export function SubmitButton({
  onClick,
  disabled,
  loading,
  loadingText = "Processing…",
  children,
}: SubmitButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-neutral-400 dark:disabled:bg-neutral-700"
    >
      {loading && (
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
      )}
      {loading ? loadingText : children}
    </button>
  );
}
