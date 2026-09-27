import { api, ApiError } from "@/lib/api";
import { KnowledgeCard } from "@/components/reasoning/KnowledgeCard";
import { ExpertSystemCard } from "@/components/reasoning/ExpertSystemCard";
import { SearchCard } from "@/components/reasoning/SearchCard";

/**
 * Tasks 1-3 — the symbolic half of the project.
 *
 * These three do not sit in the ML data pipeline, so they had no page of their
 * own until Task 9 required every task to be visible. Each card reads its real
 * artifacts through /api/reasoning/*, and says "Not generated yet" with the
 * command to run when an artifact is missing.
 */
export default async function ReasoningPage() {
  let knowledge = null;
  let expert = null;
  let search = null;
  let error: string | null = null;

  try {
    [knowledge, expert, search] = await Promise.all([
      api.knowledge(),
      api.expertSystem(),
      api.searchComparison(),
    ]);
  } catch (e) {
    error = e instanceof ApiError ? e.message : "Could not reach the backend API.";
  }

  if (error) {
    return (
      <div className="card border-red-500/30 bg-red-500/5">
        <div className="badge badge-warning">Backend unreachable</div>
        <p className="text-sm text-white/70 mt-2">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-10">
      <section>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold text-white">Reasoning</h1>
          <span className="badge">Tasks 1 – 3</span>
        </div>
        <p className="text-white/60 mt-1 max-w-3xl">
          The symbolic half of the platform: what the system <em>knows</em> (an ontology and knowledge graph),
          what it <em>concludes</em> (a forward-chaining rule base), and what it <em>plans</em> (a state-space
          search over pit strategies). Two of these feed the live strategy recommendation alongside the models.
        </p>
      </section>

      {knowledge && <KnowledgeCard data={knowledge} />}
      {expert && <ExpertSystemCard data={expert} />}
      {search && <SearchCard data={search} />}
    </div>
  );
}
