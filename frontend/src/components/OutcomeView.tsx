import type { Outcome } from "../types/outcome";

interface Props {
  outcome: Outcome;
}

const labelColor: Record<string, string> = {
  Strong: "#22c55e",
  Medium: "#eab308",
  Weak: "#dc2626",
};

export function OutcomeView({ outcome }: Props) {
  const color = labelColor[outcome.opportunity_label] || "#888";

  return (
    <div style={{ marginTop: "1.5rem" }}>
      <h2>Simulation Outcome</h2>

      {/* Score headline */}
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: "0.75rem",
          marginBottom: "0.5rem",
        }}
      >
        <span style={{ fontSize: "2rem", fontWeight: 700, color }}>
          {outcome.opportunity_score}/10
        </span>
        <span
          style={{
            fontSize: "1.1rem",
            fontWeight: 600,
            color,
            textTransform: "uppercase",
          }}
        >
          {outcome.opportunity_label}
        </span>
      </div>

      {/* Explanation */}
      {outcome.score_explanation && (
        <p style={{ color: "#555", margin: "0.25rem 0 1rem" }}>
          {outcome.score_explanation}
        </p>
      )}

      {/* Score breakdown */}
      {outcome.score_breakdown && outcome.score_breakdown.length > 0 && (
        <div style={{ marginBottom: "1rem" }}>
          <h3 style={{ fontSize: "0.95rem", marginBottom: "0.5rem" }}>
            Score Breakdown
          </h3>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "0.85rem",
            }}
          >
            <thead>
              <tr style={{ borderBottom: "2px solid #e5e7eb" }}>
                <th style={{ textAlign: "left", padding: "0.3rem 0.5rem" }}>
                  Field
                </th>
                <th style={{ textAlign: "right", padding: "0.3rem 0.5rem" }}>
                  Points
                </th>
                <th style={{ textAlign: "left", padding: "0.3rem 0.5rem" }}>
                  Reason
                </th>
              </tr>
            </thead>
            <tbody>
              {outcome.score_breakdown.map((item, i) => (
                <tr
                  key={i}
                  style={{ borderBottom: "1px solid #f3f4f6" }}
                >
                  <td style={{ padding: "0.3rem 0.5rem" }}>{item.field}</td>
                  <td
                    style={{
                      textAlign: "right",
                      padding: "0.3rem 0.5rem",
                      color: item.points > 0 ? "#22c55e" : item.points < 0 ? "#dc2626" : "#888",
                      fontWeight: 600,
                    }}
                  >
                    {item.points > 0 ? "+" : ""}
                    {item.points}
                  </td>
                  <td style={{ padding: "0.3rem 0.5rem", color: "#666" }}>
                    {item.reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Summary + recommendation */}
      <div
        style={{
          background: "#f8fafc",
          border: "1px solid #e2e8f0",
          borderRadius: "8px",
          padding: "1rem",
          marginBottom: "1rem",
        }}
      >
        <p style={{ margin: "0 0 0.5rem", fontWeight: 600 }}>Summary</p>
        <p style={{ margin: 0, color: "#444" }}>{outcome.summary}</p>
        <p style={{ margin: "0.75rem 0 0", fontWeight: 600 }}>
          Recommended Next Action
        </p>
        <p style={{ margin: 0, color: "#444" }}>
          {outcome.recommended_next_action}
        </p>
      </div>

      {/* Raw JSON fallback */}
      <details>
        <summary style={{ cursor: "pointer", color: "#888", fontSize: "0.85rem" }}>
          Raw JSON
        </summary>
        <pre
          style={{
            background: "#1e1e2e",
            color: "#cdd6f4",
            padding: "1rem",
            borderRadius: "8px",
            overflow: "auto",
            fontSize: "0.8rem",
            lineHeight: 1.5,
            marginTop: "0.5rem",
          }}
        >
          {JSON.stringify(outcome, null, 2)}
        </pre>
      </details>
    </div>
  );
}
