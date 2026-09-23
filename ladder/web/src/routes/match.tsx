import { useQueryClient } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { type FormEvent, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useCreateAutoBatch } from "@/gen/hooks/useCreateAutoBatch";
import { useCreateBatch } from "@/gen/hooks/useCreateBatch";
import { useListAgents } from "@/gen/hooks/useListAgents";
import { listBatchesQueryKey, useListBatches } from "@/gen/hooks/useListBatches";
import type { CreateAutoBatchBody } from "@/gen/types/CreateAutoBatchBody";
import type { CreateBatchBody } from "@/gen/types/CreateBatchBody";
import { getErrorMessage } from "@/lib/errors";
import { formatDate, formatNumber } from "@/lib/format";

export const Route = createFileRoute("/match")({ component: MatchPage });

function MatchPage() {
  const queryClient = useQueryClient();
  const batchesQuery = useListBatches();
  const agentsQuery = useListAgents();
  const invalidateBatches = () => queryClient.invalidateQueries({ queryKey: listBatchesQueryKey() });
  const createBatchMutation = useCreateBatch<unknown>({ mutation: { onSuccess: invalidateBatches } });
  const createAutoBatchMutation = useCreateAutoBatch<unknown>({ mutation: { onSuccess: invalidateBatches } });
  return (
    <section className="space-y-6">
      <PageHeading eyebrow="Match" title="対戦" description="Batch ごとの対戦状況と、対戦の登録を管理します。" />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Card>
          <CardHeader><CardTitle>Batch 履歴</CardTitle><CardDescription>新しい Batch から表示します。</CardDescription></CardHeader>
          <CardContent>
            {batchesQuery.isPending ? <LoadingState /> : null}
            {batchesQuery.isError ? <ErrorState message={getErrorMessage(batchesQuery.error)} /> : null}
            {!batchesQuery.isPending && !batchesQuery.isError ? <BatchTable batches={batchesQuery.data.batches ?? []} /> : null}
          </CardContent>
        </Card>
        <div className="space-y-6">
          <CreateBatchCard agents={agentsQuery.data?.agents ?? []} onSubmit={(body) => createBatchMutation.mutateAsync({ body })} pending={createBatchMutation.isPending} />
          <AutoBatchCard onSubmit={(body) => createAutoBatchMutation.mutateAsync({ body })} pending={createAutoBatchMutation.isPending} />
        </div>
      </div>
    </section>
  );
}

function BatchTable({ batches }: { batches: Array<{ id: number; status: string; games: number; done_count: number; failed_count: number; matches?: Array<unknown> | null; created_at: string; error: string }> }) {
  return <div className="overflow-x-auto"><Table><TableHeader><TableRow><TableHead>Batch</TableHead><TableHead>状態</TableHead><TableHead>Match 数</TableHead><TableHead>完了</TableHead><TableHead>失敗</TableHead><TableHead>Games</TableHead><TableHead>作成日時</TableHead></TableRow></TableHeader><TableBody>{batches.map((batch) => <TableRow key={batch.id}><TableCell><Link className="font-medium text-primary hover:underline" to="/match/$batchId" params={{ batchId: String(batch.id) }}>Batch #{batch.id}</Link></TableCell><TableCell><StatusBadge status={batch.status} /></TableCell><TableCell className="tabular-nums">{batch.matches?.length ?? "—"}</TableCell><TableCell className="tabular-nums">{formatNumber(batch.done_count)}</TableCell><TableCell className="tabular-nums">{formatNumber(batch.failed_count)}</TableCell><TableCell className="tabular-nums">{formatNumber(batch.games)}</TableCell><TableCell className="text-muted-foreground">{formatDate(batch.created_at)}</TableCell></TableRow>)}{batches.length === 0 ? <TableRow><TableCell colSpan={7} className="h-20 text-center text-muted-foreground">Batch はまだありません。</TableCell></TableRow> : null}</TableBody></Table></div>;
}

function CreateBatchCard({ agents, onSubmit, pending }: { agents: Array<{ id: number; name: string }>; onSubmit: (body: Omit<NonNullable<CreateBatchBody>, "$schema">) => Promise<unknown>; pending: boolean }) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [games, setGames] = useState("1");
  const [error, setError] = useState("");
  const submittingRef = useRef(false);
  const toggleAgent = (id: string) => setSelectedIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id]);
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const gameCount = Number(games);
    if (submittingRef.current || selectedIds.length < 2) { if (selectedIds.length < 2) setError("2つ以上の Agent を選択してください。"); return; }
    if (!Number.isSafeInteger(gameCount) || gameCount < 1) { setError("Games は1以上の整数にしてください。"); return; }
    submittingRef.current = true;
    setError("");
    try { await onSubmit({ agent_ids: selectedIds.map(Number), games: gameCount }); setSelectedIds([]); } catch (reason) { setError(getErrorMessage(reason)); } finally { submittingRef.current = false; }
  }
  return <Card><CardHeader><CardTitle>Batch を作成</CardTitle><CardDescription>2つ以上の Agent と局数を選びます。</CardDescription></CardHeader><CardContent><form className="space-y-4" onSubmit={handleSubmit}><div className="space-y-2"><p className="text-sm font-medium">Agent</p><div className="max-h-48 space-y-2 overflow-y-auto rounded-md border p-3">{agents.map((agent) => <label key={agent.id} className="flex items-center gap-2 text-sm"><Input type="checkbox" className="size-4" checked={selectedIds.includes(String(agent.id))} onChange={() => toggleAgent(String(agent.id))} /><span>{agent.name}</span></label>)}{agents.length === 0 ? <p className="text-sm text-muted-foreground">先に Agent を登録してください。</p> : null}</div></div><label className="grid gap-2 text-sm font-medium">Games<Input type="number" min={1} step={1} value={games} onChange={(event) => setGames(event.target.value)} /></label>{error ? <ErrorState message={error} /> : null}<Button type="submit" className="w-full" disabled={pending}>{pending ? "登録中…" : "Batch を登録"}</Button></form></CardContent></Card>;
}

function AutoBatchCard({ onSubmit, pending }: { onSubmit: (body: Omit<NonNullable<CreateAutoBatchBody>, "$schema">) => Promise<unknown>; pending: boolean }) {
  const [games, setGames] = useState("1");
  const [error, setError] = useState("");
  const submittingRef = useRef(false);
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const gameCount = Number(games);
    if (submittingRef.current || !Number.isSafeInteger(gameCount) || gameCount < 1) { setError("Games は1以上の整数にしてください。"); return; }
    submittingRef.current = true;
    setError("");
    try { await onSubmit({ games: gameCount }); } catch (reason) { setError(getErrorMessage(reason)); } finally { submittingRef.current = false; }
  }
  return <Card><CardHeader><CardTitle>自動 Batch</CardTitle><CardDescription>レーティングを使って組み合わせを作成します。</CardDescription></CardHeader><CardContent><form className="space-y-4" onSubmit={handleSubmit}><label className="grid gap-2 text-sm font-medium">Games<Input type="number" min={1} step={1} value={games} onChange={(event) => setGames(event.target.value)} /></label>{error ? <ErrorState message={error} /> : null}<Button type="submit" variant="outline" className="w-full" disabled={pending}>{pending ? "登録中…" : "自動 Batch を登録"}</Button></form></CardContent></Card>;
}

function StatusBadge({ status }: { status: string }) { const variant = status === "failed" ? "destructive" : status === "done" ? "secondary" : "outline"; const className = status === "partial" ? "border-amber-500/60 bg-amber-500/10 text-amber-700 dark:text-amber-300" : undefined; return <Badge variant={variant} className={className}>{status}</Badge>; }
function PageHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) { return <header><p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">{eyebrow}</p><h2 className="mt-2 text-3xl font-semibold tracking-tight">{title}</h2><p className="mt-2 text-sm text-muted-foreground">{description}</p></header>; }
function LoadingState() { return <p className="py-10 text-center text-sm text-muted-foreground">読み込み中です。</p>; }
function ErrorState({ message }: { message: string }) { return <p className="rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">{message}</p>; }
