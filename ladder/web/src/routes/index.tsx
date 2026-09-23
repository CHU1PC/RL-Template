import { Link, createFileRoute } from "@tanstack/react-router";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getErrorMessage } from "@/lib/errors";
import { formatNumber, formatPercent } from "@/lib/format";
import { useGetRanking } from "@/gen/hooks/useGetRanking";

export const Route = createFileRoute("/")({
  component: RankingPage,
});

function RankingPage() {
  const rankingQuery = useGetRanking();
  const rows = rankingQuery.data?.ranking ?? [];

  return (
    <section className="space-y-6">
      <PageHeading eyebrow="Ranking" title="ランキング" description="サーバーが返した順位をそのまま表示します。" />
      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>現在の順位</CardTitle>
          {rankingQuery.isFetching && !rankingQuery.isPending ? <Badge variant="secondary">更新中</Badge> : null}
        </CardHeader>
        <CardContent>
          {rankingQuery.isPending ? <LoadingState /> : null}
          {rankingQuery.isError ? <ErrorState message={getErrorMessage(rankingQuery.error)} /> : null}
          {!rankingQuery.isPending && !rankingQuery.isError ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>順位</TableHead>
                    <TableHead>モデル名</TableHead>
                    <TableHead className="text-right">スコア</TableHead>
                    <TableHead className="text-right">標準偏差</TableHead>
                    <TableHead className="text-right">勝率</TableHead>
                    <TableHead className="text-right">対戦数</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((row, index) => (
                    <TableRow key={row.agent_id}>
                      <TableCell className="font-medium">{index + 1}</TableCell>
                      <TableCell>
                        <Link className="font-medium text-primary hover:underline" to="/models/$agentId" params={{ agentId: String(row.agent_id) }}>
                          {row.agent}
                        </Link>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{row.mu.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{row.sigma.toFixed(1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{formatPercent(row.win_rate)}</TableCell>
                      <TableCell className="text-right tabular-nums">{formatNumber(row.games)}</TableCell>
                    </TableRow>
                  ))}
                  {rows.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} className="h-24 text-center text-muted-foreground">登録されたモデルはありません。</TableCell>
                    </TableRow>
                  ) : null}
                </TableBody>
              </Table>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </section>
  );
}

function PageHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <header>
      <p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">{eyebrow}</p>
      <h2 className="mt-2 text-3xl font-semibold tracking-tight">{title}</h2>
      <p className="mt-2 text-sm text-muted-foreground">{description}</p>
    </header>
  );
}

function LoadingState() {
  return <p className="py-10 text-center text-sm text-muted-foreground">読み込み中です。</p>;
}

function ErrorState({ message }: { message: string }) {
  return <p className="rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">{message}</p>;
}
