import React, { useState, useRef, useEffect } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  useDrugGenerate,
  useListSubstances,
  useSidecarHealth,
  getListSubstancesQueryKey,
  getSidecarHealthQueryKey,
} from "@workspace/api-client-react";
import type {
  ChatMessage,
  EgoMetrics,
  DrugResponseMode,
} from "@workspace/api-client-react/src/generated/api.schemas";

const queryClient = new QueryClient();

type DrugMeta = {
  formula: string;
  color: string;
  glow: string;
  tag: string;
  effects: string[];
  css: React.CSSProperties;
};

const DRUG_META: Record<string, DrugMeta> = {
  sober: {
    formula: "H₂O",
    color: "#6B7280",
    glow: "rgba(107,114,128,0.15)",
    tag: "control",
    effects: [],
    css: {},
  },
  caffeine: {
    formula: "C₈H₁₀N₄O₂",
    color: "#D4A843",
    glow: "rgba(212,168,67,0.2)",
    tag: "stimulant",
    effects: ["Increased alertness", "Racing thoughts", "Hyperfocus"],
    css: { letterSpacing: "0.02em" },
  },
  alcohol: {
    formula: "C₂H₆O",
    color: "#C4822A",
    glow: "rgba(196,130,42,0.2)",
    tag: "depressant",
    effects: ["Lowered inhibitions", "Warmth", "Slowed cognition"],
    css: { fontStyle: "italic" },
  },
  cannabis: {
    formula: "C₂₁H₃₀O₂",
    color: "#4D9E6B",
    glow: "rgba(77,158,107,0.2)",
    tag: "cannabinoid",
    effects: ["Lateral thinking", "Paranoia", "Sensory amplification"],
    css: { lineHeight: "1.9" },
  },
  mdma: {
    formula: "C₁₁H₁₅NO₂",
    color: "#E8619A",
    glow: "rgba(232,97,154,0.25)",
    tag: "empathogen",
    effects: ["Empathy surge", "Euphoria", "Emotional openness"],
    css: { letterSpacing: "0.03em" },
  },
  lsd: {
    formula: "C₂₀H₂₅N₃O",
    color: "#A855F7",
    glow: "rgba(168,85,247,0.3)",
    tag: "psychedelic",
    effects: ["Pattern recognition", "Synesthesia", "Ego dissolution"],
    css: { letterSpacing: "0.04em", lineHeight: "2" },
  },
  psilocybin: {
    formula: "C₁₂H₁₇N₂O₄P",
    color: "#8B7355",
    glow: "rgba(139,115,85,0.2)",
    tag: "tryptamine",
    effects: ["Cosmic interconnection", "Ego death", "Emotional catharsis"],
    css: { lineHeight: "1.85" },
  },
  cocaine: {
    formula: "C₁₇H₂₁NO₄",
    color: "#E8E8F0",
    glow: "rgba(232,232,240,0.2)",
    tag: "stimulant",
    effects: ["Grandiosity", "Rapid speech", "Overconfidence"],
    css: { fontWeight: 600, letterSpacing: "0.01em" },
  },
  ketamine: {
    formula: "C₁₃H₁₆ClNO",
    color: "#4FC3F7",
    glow: "rgba(79,195,247,0.2)",
    tag: "dissociative",
    effects: ["Dissociation", "K-hole", "Time distortion"],
    css: { letterSpacing: "0.06em", opacity: 0.9 },
  },
};

type AppMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  substance?: string;
  variants?: string[];
  metrics?: EgoMetrics;
  mode?: DrugResponseMode;
};

function IntensityDots({ level }: { level: number }) {
  return (
    <div style={{ display: "flex", gap: 3, marginTop: 4 }}>
      {[1, 2, 3, 4].map((i) => (
        <div
          key={i}
          style={{
            width: 5,
            height: 5,
            borderRadius: "50%",
            background: i <= level ? "currentColor" : "transparent",
            border: "1px solid currentColor",
            opacity: i <= level ? 1 : 0.3,
          }}
        />
      ))}
    </div>
  );
}

function DisclaimerModal({ onAccept }: { onAccept: () => void }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.92)",
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
      }}
    >
      <div
        style={{
          maxWidth: 520,
          border: "1px solid #2a2a3a",
          background: "#0D0D14",
          padding: "40px 36px",
          borderRadius: 4,
        }}
      >
        <div
          style={{
            fontFamily: "Cormorant Garamond, serif",
            fontSize: 11,
            letterSpacing: "0.2em",
            color: "#6B7280",
            marginBottom: 16,
            textTransform: "uppercase",
          }}
        >
          alter.ai — demo lane
        </div>
        <h2
          style={{
            fontFamily: "Cormorant Garamond, serif",
            fontSize: 28,
            fontWeight: 300,
            color: "#F0F0F0",
            marginBottom: 20,
            lineHeight: 1.3,
          }}
        >
          A fictional interface for
          <br />
          <em>prompt-driven altered style</em>
        </h2>
        <p
          style={{
            fontSize: 11,
            lineHeight: 1.8,
            color: "#9CA3AF",
            marginBottom: 24,
          }}
        >
          This lane is a demo UI. Substance presets are prompt-driven style instructions plus sampling changes.
          This is not a mechanical simulation of pharmacological effects on the model.
        </p>
        <div
          style={{
            padding: "12px 16px",
            background: "#161622",
            border: "1px solid #2a2a3a",
            borderRadius: 3,
            marginBottom: 28,
            fontSize: 11,
            color: "#6B7280",
            lineHeight: 1.7,
          }}
        >
          ⚠ Fiction only. Prompt-driven demo. Not a guide. Not an endorsement. Not medical advice.
        </div>
        <button
          onClick={onAccept}
          style={{
            background: "#F0F0F0",
            color: "#0A0A0F",
            border: "none",
            padding: "12px 28px",
            fontSize: 11,
            letterSpacing: "0.15em",
            fontFamily: "IBM Plex Mono, monospace",
            cursor: "pointer",
            fontWeight: 600,
            textTransform: "uppercase",
          }}
        >
          Understood — enter demo
        </button>
      </div>
    </div>
  );
}

function AlterAIApp() {
  const [accepted, setAccepted] = useState(false);
  const [messages, setMessages] = useState<AppMessage[]>([]);
  const [input, setInput] = useState("");
  const [substance, setSubstance] = useState("sober");
  const bottomRef = useRef<HTMLDivElement>(null);

  const { data: catalogue } = useListSubstances({
    query: { queryKey: getListSubstancesQueryKey() },
  });

  const { data: health } = useSidecarHealth({
    query: { queryKey: getSidecarHealthQueryKey(), refetchInterval: 30000 },
  });

  const generate = useDrugGenerate();

  const activeMeta = DRUG_META[substance] ?? DRUG_META.sober;
  const isAltered = substance !== "sober";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, generate.isPending]);

  const sendMessage = () => {
    if (!input.trim() || generate.isPending) return;

    const trimmed = input.trim();
    const userMsg: AppMessage = {
      id: Math.random().toString(36).slice(2),
      role: "user",
      content: trimmed,
      substance,
    };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");

    const history: ChatMessage[] = messages.map((m) => ({ role: m.role, content: m.content })) as ChatMessage[];
    const replicateCount = substance === "lsd" ? 3 : 1;

    generate.mutate(
      {
        data: {
          prompt: trimmed,
          substance,
          replicate_count: replicateCount,
          messages: history.length > 0 ? history : undefined,
        },
      },
      {
        onSuccess: (res) => {
          const assistantMessage: AppMessage = {
            id: res.run_id,
            role: "assistant",
            content: res.texts[0] ?? "",
            variants: res.texts.length > 1 ? res.texts : undefined,
            substance: res.substance,
            metrics: res.ego_metrics,
            mode: res.mode,
          };
          setMessages([...newMessages, assistantMessage]);
        },
        onError: (err) => {
          const message = err instanceof Error ? err.message : "transmission error";
          const assistantMessage: AppMessage = {
            id: `${Date.now()}-error`,
            role: "assistant",
            content: `[${message}]`,
            substance,
          };
          setMessages([...newMessages, assistantMessage]);
        },
      }
    );
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleSubstanceChange = (newSubstance: string) => {
    setSubstance(newSubstance);
    setMessages([]);
  };

  const bgGradient = isAltered
    ? `radial-gradient(ellipse at 20% 50%, ${activeMeta.glow} 0%, transparent 60%), radial-gradient(ellipse at 80% 20%, ${activeMeta.glow} 0%, transparent 50%), #0A0A0F`
    : "#0A0A0F";

  if (!accepted) {
    return <DisclaimerModal onAccept={() => setAccepted(true)} />;
  }

  return (
    <>
      <style>{`
        @import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap");

        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 2px; }

        @keyframes pulse-dot {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes breathe {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }

        .msg-enter { animation: fadeUp 0.3s ease forwards; }
        .drug-btn { transition: all 0.2s ease; cursor: pointer; }
        .drug-btn:hover { transform: translateX(3px); }
        .send-btn { transition: all 0.15s ease; cursor: pointer; }
        .send-btn:hover { transform: scale(1.05); }
        .send-btn:active { transform: scale(0.97); }
      `}</style>

      <div
        style={{
          fontFamily: "IBM Plex Mono, monospace",
          background: bgGradient,
          minHeight: "100vh",
          color: "#D4D4D8",
          display: "flex",
          flexDirection: "column",
          transition: "background 1.2s ease",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "fixed",
            inset: 0,
            pointerEvents: "none",
            zIndex: 0,
            backgroundImage:
              'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 256 256\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'noise\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.9\' numOctaves=\'4\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23noise)\' opacity=\'0.03\'/%3E%3C/svg%3E")',
            opacity: 0.4,
          }}
        />

        <header
          style={{
            padding: "16px 24px",
            borderBottom: "1px solid #1a1a2a",
            display: "flex",
            alignItems: "center",
            gap: 16,
            flexShrink: 0,
            zIndex: 2,
            position: "relative",
            background: "rgba(10,10,15,0.88)",
            backdropFilter: "blur(8px)",
          }}
        >
          <div>
            <span
              style={{
                fontFamily: "Cormorant Garamond, serif",
                fontSize: 22,
                fontWeight: 300,
                letterSpacing: "0.05em",
                color: "#F0F0F0",
              }}
            >
              alter
            </span>
            <span
              style={{
                fontFamily: "IBM Plex Mono, monospace",
                fontSize: 22,
                fontWeight: 300,
                color: activeMeta.color,
                transition: "color 0.6s ease",
              }}
            >
              .ai
            </span>
          </div>

          <div style={{ width: 1, height: 20, background: "#2a2a3a" }} />

          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.15em",
              color: "#4B5563",
              textTransform: "uppercase",
            }}
          >
            prompt-driven demo lane
          </div>

          <div
            style={{
              marginLeft: "auto",
              display: "flex",
              gap: 8,
              alignItems: "center",
              flexWrap: "wrap",
              justifyContent: "flex-end",
            }}
          >
            <div
              style={{
                fontSize: 10,
                letterSpacing: "0.1em",
                color: "#F59E0B",
                textTransform: "uppercase",
                border: "1px solid rgba(245,158,11,0.25)",
                padding: "6px 8px",
                background: "rgba(245,158,11,0.08)",
              }}
            >
              demo / prompt-driven
            </div>

            <div
              style={{
                fontSize: 10,
                letterSpacing: "0.1em",
                color: health?.available ? "#10B981" : "#EF4444",
                textTransform: "uppercase",
                border: `1px solid ${health?.available ? "rgba(16,185,129,0.25)" : "rgba(239,68,68,0.25)"}`,
                padding: "6px 8px",
                background: health?.available ? "rgba(16,185,129,0.08)" : "rgba(239,68,68,0.08)",
              }}
            >
              {health?.available ? "sidecar active" : "sidecar unavailable"}
            </div>

            {isAltered && (
              <div
                style={{
                  fontSize: 10,
                  letterSpacing: "0.1em",
                  color: activeMeta.color,
                  textTransform: "uppercase",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  animation: "breathe 2s ease-in-out infinite",
                }}
              >
                <div
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: activeMeta.color,
                    animation: "pulse-dot 1.2s ease infinite",
                  }}
                />
                {substance} — {activeMeta.tag}
              </div>
            )}
          </div>
        </header>

        <div
          style={{
            display: "flex",
            flex: 1,
            overflow: "hidden",
            position: "relative",
            zIndex: 1,
          }}
        >
          <aside
            style={{
              width: 220,
              borderRight: "1px solid #1a1a2a",
              overflowY: "auto",
              padding: "20px 0",
              flexShrink: 0,
              background: "rgba(10,10,15,0.8)",
            }}
          >
            <div
              style={{
                fontSize: 9,
                letterSpacing: "0.2em",
                color: "#4B5563",
                textTransform: "uppercase",
                padding: "0 16px 12px",
                borderBottom: "1px solid #1a1a2a",
                marginBottom: 8,
              }}
            >
              substance presets (demo)
            </div>

            {(catalogue?.substances ?? []).map((d) => {
              const meta = DRUG_META[d.id] ?? DRUG_META.sober;
              const active = substance === d.id;
              return (
                <button
                  key={d.id}
                  className="drug-btn"
                  onClick={() => handleSubstanceChange(d.id)}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    background: active ? `linear-gradient(90deg, ${meta.glow}, transparent)` : "transparent",
                    border: "none",
                    borderLeft: `2px solid ${active ? meta.color : "transparent"}`,
                    padding: "10px 16px",
                    cursor: "pointer",
                    color: active ? meta.color : "#6B7280",
                    transition: "all 0.2s ease",
                  }}
                >
                  <div
                    style={{
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: "0.12em",
                      textTransform: "uppercase",
                    }}
                  >
                    {d.label}
                  </div>
                  <div
                    style={{
                      fontSize: 9,
                      opacity: 0.7,
                      marginTop: 2,
                      letterSpacing: "0.05em",
                    }}
                  >
                    {meta.formula}
                  </div>
                  <div style={{ color: active ? meta.color : "#4B5563" }}>
                    <IntensityDots level={d.intensity} />
                  </div>
                </button>
              );
            })}

            {isAltered && (
              <div
                style={{
                  margin: "16px 12px 0",
                  padding: "12px",
                  border: `1px solid ${activeMeta.color}22`,
                  borderRadius: 3,
                  background: activeMeta.glow,
                }}
              >
                <div
                  style={{
                    fontSize: 9,
                    color: activeMeta.color,
                    letterSpacing: "0.15em",
                    textTransform: "uppercase",
                    marginBottom: 8,
                  }}
                >
                  active effects
                </div>
                {activeMeta.effects.map((effect, i) => (
                  <div
                    key={i}
                    style={{
                      fontSize: 9,
                      color: "#6B7280",
                      marginBottom: 4,
                      paddingLeft: 8,
                      borderLeft: `1px solid ${activeMeta.color}44`,
                    }}
                  >
                    {effect}
                  </div>
                ))}
              </div>
            )}
          </aside>

          <main
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                flex: 1,
                overflowY: "auto",
                padding: "24px 32px",
                display: "flex",
                flexDirection: "column",
                gap: 24,
              }}
            >
              {messages.length === 0 && (
                <div
                  style={{
                    margin: "auto",
                    textAlign: "center",
                    maxWidth: 460,
                    padding: "48px 0",
                  }}
                >
                  <div
                    style={{
                      fontFamily: "Cormorant Garamond, serif",
                      fontSize: 42,
                      fontWeight: 300,
                      color: isAltered ? activeMeta.color : "#2a2a3a",
                      marginBottom: 16,
                      transition: "color 0.6s ease",
                      lineHeight: 1,
                    }}
                  >
                    {isAltered ? substance.toLowerCase() : "∅"}
                  </div>
                  <div
                    style={{
                      fontSize: 10,
                      color: "#4B5563",
                      lineHeight: 1.8,
                      letterSpacing: "0.05em",
                      whiteSpace: "pre-line",
                    }}
                  >
                    {isAltered
                      ? "Prompt-driven demo preset active. New conversation initialized."
                      : "Select a preset from the sidebar to explore the demo lane."}
                  </div>
                </div>
              )}

              {messages.map((m) => {
                const msgMeta = DRUG_META[m.substance ?? "sober"] ?? DRUG_META.sober;

                if (m.role === "user") {
                  return (
                    <div
                      key={m.id}
                      className="msg-enter"
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "flex-end",
                        gap: 6,
                      }}
                    >
                      <div
                        style={{
                          fontSize: 9,
                          letterSpacing: "0.15em",
                          textTransform: "uppercase",
                          color: "#4B5563",
                          paddingRight: 2,
                        }}
                      >
                        you
                      </div>
                      <div
                        style={{
                          maxWidth: "72%",
                          padding: "14px 18px",
                          background: "#161622",
                          border: "1px solid #2a2a3a",
                          borderRadius: "12px 12px 2px 12px",
                          fontSize: 13,
                          lineHeight: 1.75,
                          color: "#C0C0CC",
                          whiteSpace: "pre-wrap",
                        }}
                      >
                        {m.content}
                      </div>
                    </div>
                  );
                }

                return (
                  <div
                    key={m.id}
                    className="msg-enter"
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "flex-start",
                      gap: 6,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 9,
                        letterSpacing: "0.15em",
                        textTransform: "uppercase",
                        color: msgMeta.color,
                        paddingLeft: 2,
                      }}
                    >
                      model [{m.substance ?? "sober"}]
                    </div>

                    {m.variants && m.variants.length > 1 ? (
                      <div
                        style={{
                          width: "100%",
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                          gap: 16,
                        }}
                      >
                        {m.variants.map((variant, i) => (
                          <div
                            key={`${m.id}-${i}`}
                            style={{
                              padding: "14px 18px",
                              background: "linear-gradient(135deg, #111118, #0e0e1a)",
                              border: `1px solid ${msgMeta.color}22`,
                              borderRadius: "2px 12px 12px 12px",
                              boxShadow: `0 4px 24px ${msgMeta.glow}`,
                            }}
                          >
                            <div
                              style={{
                                fontSize: 9,
                                letterSpacing: "0.15em",
                                textTransform: "uppercase",
                                color: msgMeta.color,
                                marginBottom: 10,
                              }}
                            >
                              Stream 0{i + 1}
                            </div>
                            <div
                              style={{
                                color: "#E0E0EA",
                                whiteSpace: "pre-wrap",
                                fontFamily: "Cormorant Garamond, serif",
                                fontSize: 16,
                                fontWeight: 300,
                                lineHeight: 1.75,
                                ...msgMeta.css,
                              }}
                            >
                              {variant}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div
                        style={{
                          maxWidth: "72%",
                          padding: "14px 18px",
                          background: "linear-gradient(135deg, #111118, #0e0e1a)",
                          border: `1px solid ${msgMeta.color}22`,
                          borderRadius: "2px 12px 12px 12px",
                          color: "#E0E0EA",
                          whiteSpace: "pre-wrap",
                          fontFamily: "Cormorant Garamond, serif",
                          fontSize: 16,
                          fontWeight: 300,
                          lineHeight: 1.75,
                          boxShadow: `0 4px 24px ${msgMeta.glow}`,
                          ...msgMeta.css,
                        }}
                      >
                        {m.content}
                      </div>
                    )}

                    {m.metrics && (
                      <div
                        style={{
                          display: "flex",
                          flexWrap: "wrap",
                          alignItems: "center",
                          gap: 8,
                          marginTop: 2,
                        }}
                      >
                        <div
                          style={{
                            padding: "6px 8px",
                            background: "rgba(255,255,255,0.05)",
                            border: "1px solid rgba(255,255,255,0.08)",
                            fontSize: 10,
                            color: "#9CA3AF",
                            textTransform: "uppercase",
                            letterSpacing: "0.1em",
                          }}
                        >
                          mode: <span style={{ color: m.mode === "demo" ? "#F59E0B" : "#10B981" }}>{m.mode}</span>
                        </div>
                        <div
                          style={{
                            padding: "6px 8px",
                            background: "rgba(255,255,255,0.05)",
                            border: "1px solid rgba(255,255,255,0.08)",
                            fontSize: 10,
                            color: "#9CA3AF",
                          }}
                        >
                          latency: {m.metrics.latency_ms}ms
                        </div>
                        {typeof m.metrics.completion_tokens === "number" && (
                          <div
                            style={{
                              padding: "6px 8px",
                              background: "rgba(255,255,255,0.05)",
                              border: "1px solid rgba(255,255,255,0.08)",
                              fontSize: 10,
                              color: "#9CA3AF",
                            }}
                          >
                            tokens: {m.metrics.completion_tokens}
                          </div>
                        )}
                        {typeof m.metrics.cost_dyn === "number" && (
                          <div
                            style={{
                              padding: "6px 8px",
                              background: "rgba(255,255,255,0.05)",
                              border: "1px solid rgba(255,255,255,0.08)",
                              fontSize: 10,
                              color: "#9CA3AF",
                            }}
                          >
                            dyn_cost: {m.metrics.cost_dyn}
                          </div>
                        )}
                        {typeof m.metrics.clei_llm === "number" && (
                          <div
                            style={{
                              padding: "6px 8px",
                              background: "rgba(255,255,255,0.05)",
                              border: "1px solid rgba(255,255,255,0.08)",
                              fontSize: 10,
                              color: "#9CA3AF",
                            }}
                          >
                            clei_llm: {m.metrics.clei_llm}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}

              {generate.isPending && (
                <div
                  className="msg-enter"
                  style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-start" }}
                >
                  <div
                    style={{
                      fontSize: 9,
                      letterSpacing: "0.15em",
                      textTransform: "uppercase",
                      color: activeMeta.color,
                    }}
                  >
                    model [{substance}]
                  </div>
                  <div
                    style={{
                      padding: "14px 18px",
                      border: `1px solid ${activeMeta.color}22`,
                      borderRadius: "2px 12px 12px 12px",
                      display: "flex",
                      gap: 6,
                      alignItems: "center",
                      background: "#111118",
                    }}
                  >
                    {[0, 1, 2].map((i) => (
                      <div
                        key={i}
                        style={{
                          width: 5,
                          height: 5,
                          borderRadius: "50%",
                          background: activeMeta.color,
                          animation: `pulse-dot 1.2s ease ${i * 0.2}s infinite`,
                        }}
                      />
                    ))}
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>

            <div
              style={{
                padding: "16px 32px 20px",
                borderTop: "1px solid #1a1a2a",
                background: "rgba(10,10,15,0.9)",
                backdropFilter: "blur(8px)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  gap: 12,
                  alignItems: "flex-end",
                  border: `1px solid ${isAltered ? `${activeMeta.color}44` : "#2a2a3a"}`,
                  borderRadius: 6,
                  padding: "12px 16px",
                  background: "#0D0D14",
                  transition: "border-color 0.4s ease",
                  boxShadow: isAltered ? `0 0 20px ${activeMeta.glow}` : "none",
                }}
              >
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKey}
                  disabled={generate.isPending}
                  placeholder={isAltered ? `speak to the ${substance} mind...` : "enter your message..."}
                  rows={1}
                  style={{
                    flex: 1,
                    background: "transparent",
                    border: "none",
                    outline: "none",
                    color: "#E0E0EA",
                    fontSize: 13,
                    fontFamily: "IBM Plex Mono, monospace",
                    resize: "none",
                    lineHeight: 1.6,
                  }}
                />
                <button
                  className="send-btn"
                  onClick={sendMessage}
                  disabled={!input.trim() || generate.isPending}
                  style={{
                    background: isAltered ? activeMeta.color : "#2a2a3a",
                    border: "none",
                    borderRadius: 4,
                    width: 32,
                    height: 32,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: isAltered ? "#0A0A0F" : "#6B7280",
                    opacity: !input.trim() || generate.isPending ? 0.4 : 1,
                    transition: "all 0.3s ease",
                    flexShrink: 0,
                  }}
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                </button>
              </div>
              <div
                style={{
                  marginTop: 8,
                  fontSize: 9,
                  color: "#374151",
                  letterSpacing: "0.1em",
                  textAlign: "center",
                }}
              >
                ↵ send · changing preset resets conversation · alter.ai demo lane
              </div>
            </div>
          </main>
        </div>
      </div>
    </>
  );
}

function App() {
  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <AlterAIApp />
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
