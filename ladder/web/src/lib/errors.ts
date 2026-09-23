const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

export function getErrorMessage(error: unknown): string {
  if (error instanceof Error && "data" in error && isRecord(error.data)) {
    const detail = error.data.detail;
    if (typeof detail === "string" && detail.length > 0) return detail;
    const title = error.data.title;
    if (typeof title === "string" && title.length > 0) return title;
  }
  if (error instanceof Error && error.message.length > 0) return error.message;
  return "サーバーとの通信に失敗しました。";
}
