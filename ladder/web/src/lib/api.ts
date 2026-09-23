import { client } from "@/gen/.kubb/client";

client.setConfig({
  baseURL: import.meta.env.VITE_API_BASE ?? "/api",
});

export const apiClient = client;
