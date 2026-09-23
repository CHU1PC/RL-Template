import { Link, Outlet, createRootRoute } from "@tanstack/react-router";
import { BarChart3, Boxes, Swords } from "lucide-react";
import type { ReactNode } from "react";

const tabs: Array<{ label: string; to: "/" | "/models" | "/match"; icon: ReactNode }> = [
  { label: "Ranking", to: "/", icon: <BarChart3 className="size-4" /> },
  { label: "Models", to: "/models", icon: <Boxes className="size-4" /> },
  { label: "Match", to: "/match", icon: <Swords className="size-4" /> },
];

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="min-h-screen bg-muted/30 text-foreground">
      <div className="mx-auto flex min-h-screen max-w-[1500px]">
        <aside className="w-56 shrink-0 border-r bg-background px-4 py-6">
          <div className="mb-8 px-2">
            <p className="text-xs font-medium uppercase tracking-[0.24em] text-muted-foreground">Private ladder</p>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight">Arena Board</h1>
          </div>
          <nav className="space-y-1" aria-label="メインメニュー">
            {tabs.map((tab) => (
              <Link
                key={tab.to}
                to={tab.to}
                activeOptions={{ exact: tab.to === "/" }}
                className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                activeProps={{ className: "bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground" }}
              >
                {tab.icon}
                {tab.label}
              </Link>
            ))}
          </nav>
        </aside>
        <main className="min-w-0 flex-1 px-6 py-8 md:px-10">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
