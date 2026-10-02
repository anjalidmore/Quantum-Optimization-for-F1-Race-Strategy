import { api, ApiError } from "@/lib/api";
import { KnowledgeCard } from "@/components/reasoning/KnowledgeCard";
import { ExpertSystemCard } from "@/components/reasoning/ExpertSystemCard";
import { SearchCard } from "@/components/reasoning/SearchCard";
import { PageHero } from "@/components/PageHero";

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
      <div className="card border-accent">
        <div className="badge badge-warning">Backend unreachable</div>
        <p className="text-sm text-paper-700 mt-2">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-10 pb-10">
      <PageHero
        photo="/images/photo-aerial-circuit.jpg"
        focus="50% 70%"
        eyebrow="Tasks 1 – 3 · the symbolic half"
        title="Reasoning"
        blurb="What the system knows (an ontology and knowledge graph), what it concludes (a forward-chaining rule base), and what it plans (a state-space search over pit strategies). Two of these feed the live strategy recommendation alongside the models."
      />

      {knowledge && <KnowledgeCard data={knowledge} />}
      {expert && <ExpertSystemCard data={expert} />}
      {search && <SearchCard data={search} />}
    </div>
  );
}
