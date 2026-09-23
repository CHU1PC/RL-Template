import { useQueryClient } from "@tanstack/react-query";
import { Link, createFileRoute } from "@tanstack/react-router";
import { type FormEvent, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useAddGroupAgent } from "@/gen/hooks/useAddGroupAgent";
import { useCreateAgent } from "@/gen/hooks/useCreateAgent";
import { useCreateGroup } from "@/gen/hooks/useCreateGroup";
import { getRankingQueryKey } from "@/gen/hooks/useGetRanking";
import { listAgentsQueryKey, useListAgents } from "@/gen/hooks/useListAgents";
import { listGroupsQueryKey, useListGroups } from "@/gen/hooks/useListGroups";
import type { CreateAgentRequestBody } from "@/gen/types/CreateAgent";
import type { CreateGroupBody } from "@/gen/types/CreateGroupBody";
import { getErrorMessage } from "@/lib/errors";

export const Route = createFileRoute("/models")({ component: ModelsPage });

function ModelsPage() {
  const queryClient = useQueryClient();
  const agentsQuery = useListAgents();
  const groupsQuery = useListGroups();
  const [assignments, setAssignments] = useState<Record<string, string[]>>({});
  const [groupDialogOpen, setGroupDialogOpen] = useState(false);
  const createAgentMutation = useCreateAgent<unknown>({
    mutation: {
      onSuccess: async () => {
        await Promise.all([
          queryClient.invalidateQueries({ queryKey: listAgentsQueryKey() }),
          queryClient.invalidateQueries({ queryKey: getRankingQueryKey() }),
        ]);
      },
    },
  });
  const createGroupMutation = useCreateGroup<unknown>({
    mutation: {
      onSuccess: async () => {
        await queryClient.invalidateQueries({ queryKey: listGroupsQueryKey() });
        setGroupDialogOpen(false);
      },
    },
  });
  const addGroupMutation = useAddGroupAgent<unknown>({
    mutation: {
      onSuccess: async (result) => {
      const groupName = result.group.name;
      const memberIds = result.agents?.map((agent) => String(agent.id)) ?? [];
      setAssignments((current) => {
        const next = { ...current };
        for (const memberId of memberIds) next[memberId] = Array.from(new Set([...(next[memberId] ?? []), groupName]));
        return next;
      });
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: listAgentsQueryKey() }),
        queryClient.invalidateQueries({ queryKey: listGroupsQueryKey() }),
      ]);
      },
    },
  });

  return (
    <section className="space-y-6">
      <PageHeading eyebrow="Models" title="モデル" description="対戦に参加する Agent と Group を管理します。" />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Card>
          <CardHeader className="flex-row items-start justify-between gap-4">
            <div><CardTitle>登録済みモデル</CardTitle><CardDescription className="mt-1">モデル名を選ぶと詳細を表示します。</CardDescription></div>
            <Dialog open={groupDialogOpen} onOpenChange={setGroupDialogOpen}>
              <DialogTrigger asChild><Button variant="outline">Group を作成</Button></DialogTrigger>
              <CreateGroupDialog onSubmit={(body) => createGroupMutation.mutateAsync({ body })} pending={createGroupMutation.isPending} />
            </Dialog>
          </CardHeader>
          <CardContent>
            {agentsQuery.isPending ? <LoadingState /> : null}
            {agentsQuery.isError ? <ErrorState message={getErrorMessage(agentsQuery.error)} /> : null}
            {!agentsQuery.isPending && !agentsQuery.isError ? (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader><TableRow><TableHead>モデル名</TableHead><TableHead>種類</TableHead><TableHead>スコア</TableHead><TableHead>標準偏差</TableHead><TableHead>Group</TableHead><TableHead className="text-right">Group に追加</TableHead></TableRow></TableHeader>
                  <TableBody>
                    {(agentsQuery.data?.agents ?? []).map((agent) => (
                      <AgentRow key={agent.id} agent={agent} groupNames={assignments[String(agent.id)] ?? []} groups={groupsQuery.data?.groups ?? []} onAdd={(groupId) => addGroupMutation.mutate({ path: { id: groupId }, body: { agent_id: agent.id } })} adding={addGroupMutation.isPending} />
                    ))}
                    {(agentsQuery.data?.agents ?? []).length === 0 ? <TableRow><TableCell colSpan={6} className="h-24 text-center text-muted-foreground">モデルはまだありません。</TableCell></TableRow> : null}
                  </TableBody>
                </Table>
              </div>
            ) : null}
            {addGroupMutation.isError ? <ErrorState message={getErrorMessage(addGroupMutation.error)} /> : null}
          </CardContent>
        </Card>
        <CreateAgentCard onSubmit={(body) => createAgentMutation.mutateAsync({ body })} pending={createAgentMutation.isPending} />
      </div>
    </section>
  );
}

function AgentRow({ agent, groupNames, groups, onAdd, adding }: {
  agent: { id: number; name: string; kind: string; rating_mu: number; rating_sigma: number };
  groupNames: string[];
  groups: Array<{ id: number; name: string }>;
  onAdd: (groupId: number) => void;
  adding: boolean;
}) {
  const [groupId, setGroupId] = useState("");
  return (
    <TableRow>
      <TableCell><Link className="font-medium text-primary hover:underline" to="/models/$agentId" params={{ agentId: String(agent.id) }}>{agent.name}</Link></TableCell>
      <TableCell><Badge variant="secondary">{agent.kind}</Badge></TableCell>
      <TableCell className="tabular-nums">{agent.rating_mu.toFixed(1)}</TableCell>
      <TableCell className="tabular-nums">{agent.rating_sigma.toFixed(1)}</TableCell>
      <TableCell><div className="flex flex-wrap gap-1">{groupNames.length > 0 ? groupNames.map((name) => <Badge key={name} variant="outline">{name}</Badge>) : <span className="text-sm text-muted-foreground">—</span>}</div></TableCell>
      <TableCell><div className="flex min-w-52 items-center justify-end gap-2"><Select value={groupId} onValueChange={setGroupId}><SelectTrigger className="h-9"><SelectValue placeholder="Group を選択" /></SelectTrigger><SelectContent>{groups.map((group) => <SelectItem key={group.id} value={String(group.id)}>{group.name}</SelectItem>)}</SelectContent></Select><Button size="sm" disabled={!groupId || adding} onClick={() => onAdd(Number(groupId))}>追加</Button></div></TableCell>
    </TableRow>
  );
}

function CreateAgentCard({ onSubmit, pending }: { onSubmit: (body: NonNullable<CreateAgentRequestBody>) => Promise<unknown>; pending: boolean }) {
  const [name, setName] = useState("");
  const [kind, setKind] = useState<"onnx" | "builtin">("builtin");
  const [builtinName, setBuiltinName] = useState("random");
  const [file, setFile] = useState<File | null>(null);
  const [formError, setFormError] = useState("");
  const submittingRef = useRef(false);
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submittingRef.current) return;
    if (!name.trim()) { setFormError("モデル名を入力してください。"); return; }
    if (kind === "onnx" && !file) { setFormError("ONNX ファイルを選択してください。"); return; }
    if (kind === "builtin" && !builtinName.trim()) { setFormError("組み込み方策名を入力してください。"); return; }
    submittingRef.current = true;
    setFormError("");
    try {
      await onSubmit({ name: name.trim(), kind, builtin_name: kind === "builtin" ? builtinName.trim() : undefined, file: kind === "onnx" ? file ?? undefined : undefined });
      setName("");
      setFile(null);
    } catch (error) { setFormError(getErrorMessage(error)); } finally { submittingRef.current = false; }
  }
  return (
    <Card>
      <CardHeader><CardTitle>新規 Agent</CardTitle><CardDescription>builtin または ONNX モデルを登録します。</CardDescription></CardHeader>
      <CardContent><form className="space-y-4" onSubmit={handleSubmit}>
        <label className="grid gap-2 text-sm font-medium">モデル名<Input value={name} onChange={(event) => setName(event.target.value)} placeholder="my-agent" /></label>
        <label className="grid gap-2 text-sm font-medium">種類<Select value={kind} onValueChange={(value) => setKind(value as "onnx" | "builtin")}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="builtin">builtin</SelectItem><SelectItem value="onnx">onnx</SelectItem></SelectContent></Select></label>
        {kind === "builtin" ? <label className="grid gap-2 text-sm font-medium">builtin 名<Input value={builtinName} onChange={(event) => setBuiltinName(event.target.value)} placeholder="random" /></label> : <label className="grid gap-2 text-sm font-medium">ONNX ファイル<Input type="file" accept=".onnx" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></label>}
        {formError ? <ErrorState message={formError} /> : null}
        <Button type="submit" className="w-full" disabled={pending}>{pending ? "登録中…" : "Agent を登録"}</Button>
      </form></CardContent>
    </Card>
  );
}

function CreateGroupDialog({ onSubmit, pending }: { onSubmit: (body: Omit<NonNullable<CreateGroupBody>, "$schema">) => Promise<unknown>; pending: boolean }) {
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const submittingRef = useRef(false);
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submittingRef.current || !name.trim()) return;
    submittingRef.current = true;
    setError("");
    try { await onSubmit({ name: name.trim() }); setName(""); } catch (reason) { setError(getErrorMessage(reason)); } finally { submittingRef.current = false; }
  }
  return <DialogContent><DialogHeader><DialogTitle>Group を作成</DialogTitle><DialogDescription>Agent をまとめるタグを作成します。</DialogDescription></DialogHeader><form className="space-y-4" onSubmit={handleSubmit}><Input value={name} onChange={(event) => setName(event.target.value)} placeholder="baseline" />{error ? <ErrorState message={error} /> : null}<DialogFooter><Button type="submit" disabled={pending}>{pending ? "作成中…" : "作成"}</Button></DialogFooter></form></DialogContent>;
}

function PageHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) { return <header><p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">{eyebrow}</p><h2 className="mt-2 text-3xl font-semibold tracking-tight">{title}</h2><p className="mt-2 text-sm text-muted-foreground">{description}</p></header>; }
function LoadingState() { return <p className="py-10 text-center text-sm text-muted-foreground">読み込み中です。</p>; }
function ErrorState({ message }: { message: string }) { return <p className="rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">{message}</p>; }
