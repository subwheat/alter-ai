import React, { useState, useEffect, useRef } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { 
  useDrugGenerate, 
  useListSubstances, 
  useSidecarHealth,
  getListSubstancesQueryKey,
  getSidecarHealthQueryKey
} from "@workspace/api-client-react";
import type { ChatMessage, EgoMetrics, DrugResponseMode, SamplingConfig } from "@workspace/api-client-react/src/generated/api.schemas";
import { Loader2, Send, Terminal, Settings2, Activity, ShieldAlert, Cpu } from "lucide-react";

const queryClient = new QueryClient();

// Static mappings
const DRUG_META: Record<string, { color: string; formula: string; effects: string }> = {
  sober: { color: "#6b7280", formula: "BASELINE", effects: "Clear thought, rational evaluation, standard logic paths." },
  caffeine: { color: "#D4A843", formula: "C₈H₁₀N₄O₂", effects: "Increased token generation speed, mild verbosity, heightened urgency." },
  alcohol: { color: "#C4822A", formula: "C₂H₆O", effects: "Reduced inhibition, degraded spelling, repetitive loops, lowered coherence." },
  cannabis: { color: "#4D9E6B", formula: "C₂₁H₃₀O₂", effects: "Tangential reasoning, philosophical drift, slow response latency." },
  mdma: { color: "#E8619A", formula: "C₁₁H₁₅NO₂", effects: "Extreme empathy, affectionate phrasing, boundary dissolution." },
  lsd: { color: "#A855F7", formula: "C₂₀H₂₅N₃O", effects: "Fractal logic, sensory metaphor overload, multimodal hallucination." },
  psilocybin: { color: "#8B7355", formula: "C₁₂H₁₇N₂O₄P", effects: "Ego dissolution, nature-centric analogies, mythic archetypes." },
  cocaine: { color: "#E8E8F0", formula: "C₁₇H₂₁NO₄", effects: "High confidence, aggressive brevity, grandiose claims." },
  ketamine: { color: "#4FC3F7", formula: "C₁₃H₁₆ClNO", effects: "Spatial dissociation, detached observation, abstract formatting." },
};

type AppMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  substance?: string;
  variants?: string[]; // For LSD multi-stream
  metrics?: EgoMetrics;
  mode?: DrugResponseMode;
};

function DisclaimerModal({ onAccept }: { onAccept: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="border border-border bg-card p-6 max-w-md w-full mx-4 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
        <div className="flex items-center gap-3 mb-4 text-white">
          <Terminal className="w-5 h-5" />
          <h2 className="text-lg font-bold tracking-widest uppercase">alter.ai</h2>
        </div>
        <div className="space-y-4 text-sm text-muted-foreground leading-relaxed">
          <p>
            RESEARCH SIMULATION TERMINAL.
          </p>
          <p>
            This interface provides a fictional experiment in altered cognition via language model perturbation.
          </p>
          <p className="text-destructive font-semibold">
            NOT MEDICAL ADVICE. NOT FOR CLINICAL USE.
          </p>
          <p>
            By entering, you acknowledge the experimental nature of this bridge.
          </p>
        </div>
        <div className="mt-8 flex justify-end">
          <button 
            onClick={onAccept}
            className="px-6 py-2 bg-white text-black font-bold uppercase tracking-widest text-xs hover:bg-gray-200 transition-colors"
          >
            Acknowledge & Enter
          </button>
        </div>
      </div>
    </div>
  );
}

function AlterAIApp() {
  const [accepted, setAccepted] = useState(false);
  const [substance, setSubstance] = useState("sober");
  const [messages, setMessages] = useState<AppMessage[]>([]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { data: catalogue } = useListSubstances({ 
    query: { queryKey: getListSubstancesQueryKey() } 
  });
  
  const { data: health } = useSidecarHealth({
    query: { queryKey: getSidecarHealthQueryKey(), refetchInterval: 30000 }
  });

  const generate = useDrugGenerate();

  const activeColor = DRUG_META[substance]?.color || DRUG_META.sober.color;
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubstanceChange = (newSubstance: string) => {
    setSubstance(newSubstance);
    setMessages([]); // Reset conversation on substance change
  };

  const handleSend = async () => {
    if (!input.trim() || generate.isPending) return;

    const userMessage: AppMessage = {
      id: Math.random().toString(36).substring(7),
      role: 'user',
      content: input,
      substance
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput("");

    const isLSD = substance === 'lsd';
    
    // Prepare history for API
    const history: ChatMessage[] = messages.flatMap(m => {
      // if variant, pick first or something? Just use content.
      return { role: m.role, content: m.content } as ChatMessage;
    });

    try {
      generate.mutate({
        data: {
          prompt: currentInput,
          substance,
          replicate_count: isLSD ? 3 : 1,
          messages: history.length > 0 ? history : undefined
        }
      }, {
        onSuccess: (res) => {
          const assistantMessage: AppMessage = {
            id: res.run_id,
            role: 'assistant',
            content: res.texts[0] || "",
            variants: isLSD ? res.texts : undefined,
            substance: res.substance,
            metrics: res.ego_metrics,
            mode: res.mode
          };
          setMessages(prev => [...prev, assistantMessage]);
        }
      });
    } catch (e) {
      console.error(e);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!accepted) {
    return <DisclaimerModal onAccept={() => setAccepted(true)} />;
  }

  const activeMeta = DRUG_META[substance];

  return (
    <div 
      className="min-h-screen w-full bg-black text-gray-300 flex flex-col font-mono selection:bg-white/20"
      style={{
        background: `radial-gradient(circle at 50% 50%, ${activeColor}15 0%, transparent 60%)`,
        transition: 'background 1s ease-in-out'
      }}
    >
      {/* Header */}
      <header className="h-12 border-b border-white/10 flex items-center justify-between px-4 z-10 bg-black/40 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-white/50" />
          <span className="font-bold tracking-widest text-xs uppercase text-white/80">alter.ai</span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          {health?.available ? (
            <div className="flex items-center gap-2 text-white/50 bg-white/5 px-2 py-1 rounded">
              <Cpu className="w-3 h-3 text-green-500" />
              <span>SIDECAR ACTIVE</span>
              {health.model && <span className="opacity-50">[{health.model}]</span>}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-white/50 bg-white/5 px-2 py-1 rounded">
              <Activity className="w-3 h-3 text-yellow-500" />
              <span>DEMO MODE</span>
            </div>
          )}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 border-r border-white/10 flex flex-col bg-black/40 backdrop-blur-sm z-10 overflow-y-auto">
          <div className="p-4 border-b border-white/5">
            <h3 className="text-[10px] tracking-widest uppercase text-white/40 mb-1">Substance Selector</h3>
            <p className="text-xs text-white/30 leading-tight">Changing parameter resets current session context.</p>
          </div>
          <div className="flex-1 p-2 space-y-1">
            {catalogue?.substances.map(sub => {
              const meta = DRUG_META[sub.id] || DRUG_META.sober;
              const isActive = substance === sub.id;
              
              return (
                <button
                  key={sub.id}
                  onClick={() => handleSubstanceChange(sub.id)}
                  className={`w-full text-left p-3 rounded transition-all duration-300 relative overflow-hidden group ${isActive ? 'bg-white/5' : 'hover:bg-white/5'}`}
                >
                  {isActive && (
                    <div 
                      className="absolute left-0 top-0 bottom-0 w-1 shadow-[0_0_10px_currentColor]"
                      style={{ backgroundColor: meta.color, color: meta.color }}
                    />
                  )}
                  <div className="flex justify-between items-center mb-1">
                    <span 
                      className={`font-bold tracking-wider uppercase text-sm ${isActive ? '' : 'text-white/60 group-hover:text-white/80'}`}
                      style={{ color: isActive ? meta.color : undefined, textShadow: isActive ? `0 0 10px ${meta.color}40` : 'none' }}
                    >
                      {sub.label}
                    </span>
                    <div className="flex gap-0.5">
                      {[1,2,3,4].map(i => (
                        <div 
                          key={i} 
                          className={`w-1.5 h-1.5 rounded-full ${i <= sub.intensity ? 'bg-current opacity-80' : 'bg-white/10'}`}
                          style={{ color: i <= sub.intensity && isActive ? meta.color : undefined }}
                        />
                      ))}
                    </div>
                  </div>
                  <div className="text-[10px] text-white/30 font-mono opacity-80">
                    {meta.formula} • {sub.family}
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col relative min-w-0">
          
          {/* Active Effects Banner */}
          {substance !== 'sober' && activeMeta && (
            <div 
              className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-2 text-xs border bg-black/80 backdrop-blur z-20 flex items-center gap-3 animate-in slide-in-from-top-4 fade-in"
              style={{ borderColor: `${activeMeta.color}40`, boxShadow: `0 0 20px ${activeMeta.color}10` }}
            >
              <Settings2 className="w-4 h-4 animate-spin-slow" style={{ color: activeMeta.color }} />
              <div>
                <span className="text-white/40">ACTIVE EFFECTS // </span>
                <span style={{ color: activeMeta.color }} className="opacity-90">{activeMeta.effects}</span>
              </div>
            </div>
          )}

          {/* Chat Scroll Area */}
          <div className="flex-1 overflow-y-auto p-4 md:p-8 pb-32">
            <div className="max-w-3xl mx-auto space-y-12">
              {messages.length === 0 && (
                <div className="h-full min-h-[50vh] flex items-center justify-center opacity-20">
                  <div className="text-center space-y-4">
                    <Terminal className="w-12 h-12 mx-auto" />
                    <p className="text-sm tracking-widest uppercase">Awaiting input sequence...</p>
                  </div>
                </div>
              )}
              
              {messages.map((msg, i) => {
                const isUser = msg.role === 'user';
                const msgMeta = DRUG_META[msg.substance || 'sober'] || DRUG_META.sober;
                
                if (isUser) {
                  return (
                    <div key={msg.id} className="flex justify-end">
                      <div className="max-w-[80%] bg-white/5 border border-white/10 p-4 text-sm leading-relaxed text-white/80 rounded-sm">
                        {msg.content}
                      </div>
                    </div>
                  );
                }

                // Assistant message
                return (
                  <div key={msg.id} className="flex flex-col gap-2 max-w-[90%]">
                    {msg.variants && msg.variants.length > 1 ? (
                      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        {msg.variants.map((v, vi) => (
                          <div 
                            key={vi} 
                            className="border p-5 relative bg-black/60 shadow-2xl transition-all hover:bg-black"
                            style={{ 
                              borderColor: `${msgMeta.color}40`,
                              boxShadow: `0 0 30px ${msgMeta.color}05 inset`
                            }}
                          >
                            <div 
                              className="absolute top-0 right-0 px-2 py-0.5 text-[9px] border-b border-l tracking-widest uppercase"
                              style={{ borderColor: `${msgMeta.color}40`, color: msgMeta.color }}
                            >
                              Stream 0{vi + 1}
                            </div>
                            <div className="font-serif text-lg md:text-xl leading-relaxed text-white/90 whitespace-pre-wrap pt-2">
                              {v}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div 
                        className="border p-6 md:p-8 relative bg-black/60 shadow-2xl"
                        style={{ 
                          borderColor: `${msgMeta.color}50`,
                          boxShadow: `0 0 40px ${msgMeta.color}10 inset, 0 0 20px ${msgMeta.color}10`
                        }}
                      >
                        <div className="font-serif text-lg md:text-xl lg:text-2xl leading-relaxed text-white/90 whitespace-pre-wrap">
                          {msg.content}
                        </div>
                      </div>
                    )}
                    
                    {/* Metrics Footer */}
                    {msg.metrics && (
                      <div className="flex flex-wrap items-center gap-2 mt-2">
                        <div className="px-2 py-1 bg-white/5 border border-white/10 text-[10px] text-white/50 uppercase tracking-wider flex items-center gap-1">
                          <Activity className="w-3 h-3" />
                          MODE: <span className={msg.mode === 'demo' ? 'text-yellow-500' : 'text-green-500'}>{msg.mode}</span>
                        </div>
                        {msg.metrics.latency_ms && (
                          <div className="px-2 py-1 bg-white/5 border border-white/10 text-[10px] text-white/50 font-mono">
                            LATENCY: {msg.metrics.latency_ms}ms
                          </div>
                        )}
                        {msg.metrics.completion_tokens && (
                          <div className="px-2 py-1 bg-white/5 border border-white/10 text-[10px] text-white/50 font-mono">
                            TOKENS: {msg.metrics.completion_tokens}
                          </div>
                        )}
                        {msg.metrics.cost_dyn && (
                          <div className="px-2 py-1 bg-white/5 border border-white/10 text-[10px] text-white/50 font-mono">
                            DYN_COST: {msg.metrics.cost_dyn.toFixed(4)}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}

              {generate.isPending && (
                <div className="flex max-w-[90%]">
                  <div 
                    className="border p-6 bg-black/60 shadow-2xl flex items-center gap-3 text-white/50"
                    style={{ borderColor: `${activeMeta.color}30` }}
                  >
                    <Loader2 className="w-5 h-5 animate-spin" style={{ color: activeMeta.color }} />
                    <span className="text-xs uppercase tracking-widest font-mono">Synthesizing response...</span>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <div className="absolute bottom-0 left-0 w-full p-4 bg-gradient-to-t from-black via-black/90 to-transparent">
            <div className="max-w-3xl mx-auto relative group">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={generate.isPending}
                placeholder="Enter prompt sequence..."
                className="w-full bg-black/80 border border-white/20 p-4 pr-14 rounded-sm text-sm text-white resize-none focus:outline-none focus:border-white/40 transition-colors disabled:opacity-50 min-h-[60px] max-h-[200px]"
                style={{ 
                  boxShadow: input.trim() ? `0 0 15px ${activeMeta.color}20` : 'none',
                  borderColor: input.trim() ? `${activeMeta.color}50` : undefined
                }}
                rows={Math.min(5, input.split('\n').length || 1)}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || generate.isPending}
                className="absolute right-2 bottom-2 p-2 rounded text-white/50 hover:text-white disabled:opacity-30 transition-all"
                style={{ 
                  color: input.trim() ? activeMeta.color : undefined,
                  textShadow: input.trim() ? `0 0 10px ${activeMeta.color}80` : 'none'
                }}
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
          
        </main>
      </div>

      {/* Footer */}
      <footer className="h-8 border-t border-white/10 flex items-center justify-center text-[9px] text-white/30 uppercase tracking-[0.2em] bg-black z-10">
        alter.ai v0.1 — ego→llm bridge
      </footer>
    </div>
  );
}

function App() {
  // force dark mode on body
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
