import { defineConfig } from "kubb";
import { pluginTs } from "@kubb/plugin-ts";
import { pluginZod } from "@kubb/plugin-zod";
import { pluginFetch } from "@kubb/plugin-fetch";
import { pluginReactQuery } from "@kubb/plugin-react-query";
import type { Macro } from "@kubb/ast";

const toPascalCase = (value: string) =>
  value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join("");

const normalizeInt64: Macro = {
  name: "normalize-int64",
  property: (node) =>
    node.schema.type === "bigint"
      ? { ...node, schema: { ...node.schema, type: "integer" } }
      : undefined,
  schema: (node) => (node.type === "bigint" ? { ...node, type: "integer" } : undefined),
};

export default defineConfig({
  input: "../openapi.yaml",
  output: { path: "./src/gen", clean: true },
  plugins: [
    pluginTs({
      output: { path: "types", barrel: false },
      resolver: {
        response: {
          body: (node) => `${toPascalCase(node.operationId)}RequestBody`,
        },
      },
      macros: [
        normalizeInt64,
      ],
    }),
    pluginZod({
      output: { path: "zod", barrel: false },
      resolver: {
        response: {
          body: (node) => `${toPascalCase(node.operationId).replace(/^[A-Z]/, (char) => char.toLowerCase())}RequestBodySchema`,
        },
      },
      macros: [normalizeInt64],
    }),
    pluginFetch(),
    pluginReactQuery({ client: "fetch", hooks: true }),
  ],
});
