import { Link, createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useGetBatch } from "@/gen/hooks/useGetBatch";
import { useGetMatch } from "@/gen/hooks/useGetMatch";
import { getErrorMessage } from "@/lib/errors";
import { formatDate, formatNumber, formatPercent } from "@/lib/format";

export const Route = createFileRoute("/match/$batchId")({ component: BatchDetailPage });

function BatchDetailPage() {
  const { batchId: rawBatchId } = Route.useParams();
  const batchId = parseId(rawBatchId);
  const [selectedMatchId, setSelectedMatchId] = useState<number | null>(null);
  const batchQuery = useGetBatch(
    { path: { id: batchId ?? 0 } },
    {
      query: {
        enabled: batchId !== null,
        refetchInterval: (query) => {
          const status = query.state.data?.status;
          return status === "pending" || status === "running" ? 2000 : false;
        },
      },
    },
  );
  const matches = batchQuery.data?.matches ?? [];
  const selectedMatch = matches.find((match) => match.id === selectedMatchId) ?? matches[0];
  const matchQuery = useGetMatch(
    { path: { id: selectedMatch?.id ?? 0 } },
    { query: { enabled: selectedMatch !== undefined } },
  );
  if (batchId === null) return <ErrorState message="Batch ID が不正です。" />;
  if (batchQuery.isPending) return <LoadingState />;
  if (batchQuery.isError) return <ErrorState message={getErrorMessage(batchQuery.error)} />;
  const batch = batchQuery.data;

  return (
    <section className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4"><div><p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">Match detail</p><h2 className="mt-2 text-3xl font-semibold tracking-tight">Batch #{batch.id}</h2><p className="mt-2 text-sm text-muted-foreground">{formatDate(batch.created_at)}</p></div><Link className="text-sm font-medium text-primary hover:underline" to="/match">対戦一覧へ戻る</Link></header>
      <div className="flex flex-wrap items-center gap-3"><StatusBadge status={batch.status} /><span className="text-sm text-muted-foreground">{formatNumber(batch.games)} games</span><span className="text-sm text-muted-foreground">完了 {formatNumber(batch.done_count)} / 失敗 {formatNumber(batch.failed_count)}</span></div>
      {batch.status === "failed" && batch.error ? <ErrorState message={batch.error} /> : null}
      <Card><CardHeader><CardTitle>Matches</CardTitle></CardHeader><CardContent><MatchTable matches={matches} selectedMatchId={selectedMatch?.id ?? null} onSelect={setSelectedMatchId} /></CardContent></Card>
      <Card><CardHeader><CardTitle>Games {selectedMatch ? `— ${selectedMatch.agent_a_name} vs ${selectedMatch.agent_b_name}` : ""}</CardTitle></CardHeader><CardContent>{selectedMatch === undefined ? <p className="text-sm text-muted-foreground">Match はまだありません。</p> : matchQuery.isPending ? <LoadingState /> : matchQuery.isError ? <ErrorState message={getErrorMessage(matchQuery.error)} /> : <GameTable games={matchQuery.data.games ?? []} />}</CardContent></Card>
    </section>
  );
}

function MatchTable({ matches, selectedMatchId, onSelect }: { matches: Array<{ id: number; agent_a_name: string; agent_b_name: string; games: number; win_a: number; draw: number; loss_a: number; status: string; error: string }>; selectedMatchId: number | null; onSelect: (id: number) => void }) {
  return <div className="overflow-x-auto"><Table><TableHeader><TableRow><TableHead>対戦</TableHead><TableHead>win-draw-lose</TableHead><TableHead>A の勝率</TableHead><TableHead>状態</TableHead></TableRow></TableHeader><TableBody>{matches.map((match) => <TableRow key={match.id} className={selectedMatchId === match.id ? "bg-muted/70" : ""}><TableCell><Button variant="link" className="h-auto p-0 font-medium" onClick={() => onSelect(match.id)}>{match.agent_a_name} vs {match.agent_b_name}</Button></TableCell><TableCell className="tabular-nums">{match.win_a}-{match.draw}-{match.loss_a}</TableCell><TableCell className="tabular-nums">{match.games > 0 ? formatPercent(match.win_a / match.games) : "—"}</TableCell><TableCell><div className="space-y-1"><StatusBadge status={match.status} />{match.status === "failed" && match.error ? <p className="text-xs text-destructive">{match.error}</p> : null}</div></TableCell></TableRow>)}{matches.length === 0 ? <TableRow><TableCell colSpan={4} className="h-20 text-center text-muted-foreground">Match はまだありません。</TableCell></TableRow> : null}</TableBody></Table></div>;
}

function GameTable({ games }: { games: Array<{ id: number; index: number; result: string; turns: number; duration_ms: number }> }) {
  return <div className="overflow-x-auto"><Table><TableHeader><TableRow><TableHead>index</TableHead><TableHead>result</TableHead><TableHead>turns</TableHead><TableHead>duration</TableHead><TableHead>replay</TableHead></TableRow></TableHeader><TableBody>{games.map((game) => <TableRow key={game.id}><TableCell className="tabular-nums">{game.index}</TableCell><TableCell>{game.result}</TableCell><TableCell className="tabular-nums">{game.turns}</TableCell><TableCell className="tabular-nums">{game.duration_ms} ms</TableCell><TableCell><Button variant="outline" size="sm" disabled>Replay</Button></TableCell></TableRow>)}{games.length === 0 ? <TableRow><TableCell colSpan={5} className="h-20 text-center text-muted-foreground">Game はまだありません。</TableCell></TableRow> : null}</TableBody></Table></div>;
}

// TODO: replay viewer is the only competition-specific part of the web app.
function StatusBadge({ status }: { status: string }) { const variant = status === "failed" ? "destructive" : status === "done" ? "secondary" : "outline"; const className = status === "partial" ? "border-amber-500/60 bg-amber-500/10 text-amber-700 dark:text-amber-300" : undefined; return <Badge variant={variant} className={className}>{status}</Badge>; }
function parseId(value: string): number | null { const parsed = Number(value); return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null; }
function LoadingState() { return <p className="py-10 text-center text-sm text-muted-foreground">読み込み中です。</p>; }
function ErrorState({ message }: { message: string }) { return <p className="rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">{message}</p>; }
