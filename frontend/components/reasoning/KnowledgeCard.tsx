import { artifactUrl, KnowledgeSummary, Unavailable } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { NotGenerated } from "@/components/reasoning/NotGenerated";

/** Task 1 — counts come from the schema the ontology and graph were built from. */
export function KnowledgeCard({ data }: { data: KnowledgeSummary | Unavailable }) {
  if (!data.available) return <NotGenerated title="Task 1 — Knowledge representation" reason={data.reason} />;

  return (
    <section>
      <h2 className="text-lg font-semibold text-white mb-1">Task 1 — Knowledge representation</h2>
      <p className="text-sm text-white/50 mb-3">
        An F1 domain model: entities with typed attributes, named relationships between them, an OWL ontology
        and a populated knowledge graph.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="card">
          <div className="stat-label">Entities</div>
          <div className="stat-value">{data.n_entities}</div>
        </div>
        <div className="card">
          <div className="stat-label">Relationships</div>
          <div className="stat-value">{data.n_relationships}</div>
        </div>
        <div className="card">
          <div className="stat-label">Attributes</div>
          <div className="stat-value">{data.n_attributes}</div>
        </div>
        <div className="card">
          <div className="stat-label">Categories</div>
          <div className="stat-value">{data.categories.length}</div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="card">
          <h3 className="font-semibold text-white mb-2 text-sm">Entities by category</h3>
          <table className="w-full text-sm">
            <tbody className="text-white/80">
              {Object.entries(data.entities_by_category).map(([category, n]) => (
                <tr key={category} className="border-t border-white/5">
                  <td className="py-1">{category}</td>
                  <td className="text-right tabular-nums">{n}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-white/40 mt-3">
            Serialised as{" "}
            {Object.entries(data.files)
              .filter(([, present]) => present)
              .map(([name]) => name.replace(/_/g, " "))
              .join(", ") || "no files yet"}
            .
          </p>
        </div>
        <div className="card">
          <ArtifactImage
            src={artifactUrl("knowledge_representation/diagrams/schema_graph.png")}
            alt="Ontology schema graph"
            className="w-full rounded"
          />
          <p className="text-xs text-white/50 mt-2">The schema: entity types and the relationships between them.</p>
        </div>
      </div>
    </section>
  );
}
