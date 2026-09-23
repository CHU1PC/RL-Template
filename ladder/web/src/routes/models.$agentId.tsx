import { Link, createFileRoute } from "@tanstack/react-router";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useGetAgent } from "@/gen/hooks/useGetAgent";
import { useListMatches } from "@/gen/hooks/useListMatches";
import { getErrorMessage } from "@/lib/errors";
import { formatDate, formatPercent } from "@/lib/format";

export const Route = createFileRoute("/models/$agentId")({ component: ModelDetailPage });

function ModelDetailPage() {
  const { agentId: rawAgentId } = Route.useParams();
  const agentId = parseId(rawAgentId);
  const agentQuery = useGetAgent(
    { path: { id: agentId ?? 0 } },
    { query: { enabled: agentId !== null } },
  );
  const matchesQuery = useListMatches(
    { query: agentId === null ? undefined : { agent_id: agentId } },
    { query: { enabled: agentId !== null } },
  );
  if (agentId === null) return <ErrorState message="モデル ID が不正です。" />;
  if (agentQuery.isPending || matchesQuery.isPending) return <LoadingState />;
  if (agentQuery.isError) return <ErrorState message={getErrorMessage(agentQuery.error)} />;
  if (matchesQuery.isError) return <ErrorState message={getErrorMessage(matchesQuery.error)} />;
  const agent = agentQuery.data;
  const matches = matchesQuery.data.matches ?? [];
  const records = buildRecords(matches, agentId);
  const totals = Array.from(records.values()).reduce((total, record) => ({
    win: total.win + record.win,
    draw: total.draw + record.draw,
    loss: total.loss + record.loss,
  }), { win: 0, draw: 0, loss: 0 });
  const games = totals.win + totals.draw + totals.loss;

  return (
    <section className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div><p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">Model detail</p><h2 className="mt-2 text-3xl font-semibold tracking-tight">{agent.name}</h2><p className="mt-2 text-sm text-muted-foreground">{agent.kind}</p></div>
        <Link className="text-sm font-medium text-primary hover:underline" to="/models">モデル一覧へ戻る</Link>
      </header>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        <Metric label="スコア" value={agent.rating_mu.toFixed(1)} />
        <Metric label="標準偏差" value={agent.rating_sigma.toFixed(1)} />
        <Metric label="対戦数" value={String(games)} />
        <Metric label="勝率" value={games > 0 ? formatPercent(totals.win / games) : "—"} />
        <Metric label="記録" value={`${totals.win}-${totals.draw}-${totals.loss}`} />
      </div>
      <Card><CardHeader><CardTitle>対戦相手別の成績</CardTitle></CardHeader><CardContent><RecordTable records={records} /></CardContent></Card>
      <Card><CardHeader><CardTitle>対戦履歴</CardTitle></CardHeader><CardContent><HistoryTable matches={matches} agentId={agentId} /></CardContent></Card>
    </section>
  );
}

function buildRecords(matches: Array<{ agent_a_id: number; agent_a_name: string; agent_b_id: number; agent_b_name: string; win_a: number; draw: number; loss_a: number }>, agentId: number) {
  const records = new Map<string, { name: string; win: number; draw: number; loss: number }>();
  for (const match of matches) {
    const isA = match.agent_a_id === agentId;
    const opponentId = isA ? match.agent_b_id : match.agent_a_id;
    const current = records.get(String(opponentId)) ?? { name: isA ? match.agent_b_name : match.agent_a_name, win: 0, draw: 0, loss: 0 };
    current.win += isA ? match.win_a : match.loss_a;
    current.draw += match.draw;
    current.loss += isA ? match.loss_a : match.win_a;
    records.set(String(opponentId), current);
  }
  return records;
}

function RecordTable({ records }: { records: Map<string, { name: string; win: number; draw: number; loss: number }> }) {
  return <div className="overflow-x-auto"><Table><TableHeader><TableRow><TableHead>対戦相手</TableHead><TableHead>win-draw-lose</TableHead><TableHead>勝率</TableHead></TableRow></TableHeader><TableBody>{Array.from(records.entries()).map(([id, record]) => { const games = record.win + record.draw + record.loss; return <TableRow key={id}><TableCell className="font-medium">{record.name}</TableCell><TableCell className="tabular-nums">{record.win}-{record.draw}-{record.loss}</TableCell><TableCell className="tabular-nums">{games > 0 ? formatPercent(record.win / games) : "—"}</TableCell></TableRow>; })}{records.size === 0 ? <TableRow><TableCell colSpan={3} className="h-20 text-center text-muted-foreground">対戦履歴はありません。</TableCell></TableRow> : null}</TableBody></Table></div>;
}

function HistoryTable({ matches, agentId }: { matches: Array<{ id: number; batch_id: number; agent_a_id: number; agent_a_name: string; agent_b_name: string; win_a: number; draw: number; loss_a: number; status: string; created_at: string }>; agentId: number }) {
  return <div className="overflow-x-auto"><Table><TableHeader><TableRow><TableHead>対戦</TableHead><TableHead>結果</TableHead><TableHead>状態</TableHead><TableHead>日時</TableHead></TableRow></TableHeader><TableBody>{matches.map((match) => { const isA = match.agent_a_id === agentId; const record = isA ? `${match.win_a}-${match.draw}-${match.loss_a}` : `${match.loss_a}-${match.draw}-${match.win_a}`; return <TableRow key={match.id}><TableCell><Link className="text-primary hover:underline" to="/match/$batchId" params={{ batchId: String(match.batch_id) }}>{match.agent_a_name} vs {match.agent_b_name}</Link></TableCell><TableCell className="tabular-nums">{record}</TableCell><TableCell><Badge variant="outline">{match.status}</Badge></TableCell><TableCell className="text-muted-foreground">{formatDate(match.created_at)}</TableCell></TableRow>; })}{matches.length === 0 ? <TableRow><TableCell colSpan={4} className="h-20 text-center text-muted-foreground">対戦履歴はありません。</TableCell></TableRow> : null}</TableBody></Table></div>;
}

function Metric({ label, value }: { label: string; value: string }) { return <Card><CardContent className="pt-6"><p className="text-sm text-muted-foreground">{label}</p><p className="mt-2 text-2xl font-semibold tabular-nums">{value}</p></CardContent></Card>; }
function parseId(value: string): number | null { const parsed = Number(value); return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null; }
function LoadingState() { return <p className="py-10 text-center text-sm text-muted-foreground">読み込み中です。</p>; }
function ErrorState({ message }: { message: string }) { return <p className="rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">{message}</p>; }
