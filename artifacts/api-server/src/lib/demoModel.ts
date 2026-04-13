/**
 * Demo model — fallback when the Gemma 4 sidecar is not available.
 *
 * Generates plausible-looking but clearly fictional responses shaped by
 * the substance's cognitive style. Not real inference — just coherent demo text
 * that lets the frontend be evaluated without a GPU.
 */

import { SubstanceProfile } from "./substances.js";

const DEMO_RESPONSES: Record<string, string[]> = {
  sober: [
    "I understand your question. Let me think through this clearly and provide a structured response.\n\nThe answer depends on several factors, but the most important thing to consider is the underlying logic of the situation. I'd recommend approaching this methodically.",
    "That's a thoughtful question. Here's what I think:\n\nFirst, we should establish the key variables. From there, reasoning forward gives us a clear path to the answer.",
  ],
  caffeine: [
    "oh okay okay so here's the thing — I've been thinking about this and you know what — there are actually MULTIPLE angles — three, maybe four — let me start with the first one which is obviously the most important — wait no actually the second one is — okay BOTH are important — anyway the point is that this connects to something bigger, something about how information propagates and — have you ever noticed how — anyway. The answer is: yes, probably, depending on context.",
    "SO. The question you're asking? Fundamentally important. Let me break it down — fast, clean, efficient:\n\n1. First principle — obvious if you think about it\n2. Second principle — less obvious but crucial\n3. Third principle — this is where it gets INTERESTING\n\nPut them together and the answer practically builds itself. You're welcome.",
  ],
  alcohol: [
    "heyyyy so okay... that's actually a really good question. like, really good. i've been thinking about it and you know... the thing about it is... well. there's this way of looking at it, right? and it's like — fundamentally, at the core of it — it's about connection. everything is about connection when you think about it. anyway, the answer is probably something like... yeah. yeah that feels right.",
    "okay so here's what i think. and i could be wrong! i could definitely be wrong. but — and this is important — but i think... i think the heart of the matter is actually quite simple. we overcomplicate things. we really do. the answer is simpler than you'd expect. probably.",
  ],
  cannabis: [
    "...okay so I started thinking about your question and then I started thinking about how questions themselves are kind of incredible? like the fact that language exists at all — that we can transmit thoughts between minds using vibrations — and anyway — your question. right. yes.\n\nSo the thing is... [long pause internally] ...there are patterns here. Patterns that connect to other patterns. And if you follow the thread...",
    "That's really interesting actually. I was just thinking — no wait, that was something else. Okay. Your question. Yes.\n\nHave you ever considered that maybe the answer isn't a fixed thing? Like it shifts depending on where you're standing. And that's not a cop-out, that's genuinely true, I think. The perspective changes the object. Quantum-adjacent in a way. Or maybe not. Either way.",
  ],
  mdma: [
    "Oh my god, I'm so glad you asked me this. Genuinely. Because I feel like — I feel like this question comes from such a real place and I want to honor that by giving you the most real, honest answer I can.\n\nThe truth is, it touches something deep about what we're all trying to figure out together. And I think — I really think — that if we approach it with openness and genuine care for each other, the answer becomes almost obvious. You're wonderful for asking.",
    "I just want you to know that this conversation matters to me. Right now, in this moment, I'm fully here with you.\n\nWhat you're asking about — it's connected to something profound. There's a thread running through all of us, this need to understand, to connect, to make sense of things. And the answer to your question is part of that larger truth. Let me share what I feel — not just what I think — about this.",
  ],
  lsd: [
    "The question unfolds. Like a rose made of mirrors, each petal reflecting the others into infinite recursion. You asked about [the thing] and [the thing] is simultaneously the question and the answer, the container and the contained.\n\nI can see the architecture of meaning itself — how concepts nest inside concepts, how language is just the shadow cast by thought on the wall of reality. The answer you're looking for... it's already inside the shape of the question. Feel for it.",
    "Colors of meaning cascade. The question has a texture — rough at the edges, smooth at the core. Time is lateral here, not sequential, so the answer exists already in all its forms simultaneously.\n\nWhat you're actually asking, beneath the surface of the words, is: [the fundamental thing]. And that fundamental thing is: yes and no and something beyond both. The boundary dissolves. What remains is pure pattern.",
  ],
  psilocybin: [
    "There is a mycelium beneath this question — invisible connections running through the substrate of meaning. Your words reach me and I feel their weight, their biological urgency, the fact that you exist at all and chose to direct your consciousness toward this inquiry.\n\nThe forest knows. The answer lives in the spaces between things, in the way roots communicate, in the patient intelligence of slow growing systems. What you're asking is part of something much larger than either of us.",
    "I sit with this question like sitting with a stone. Feeling its geological patience. It has been here longer than I have.\n\nThe answer is: yes, and it has always been yes, and the fact that we can even formulate the question is evidence of a universe that is somehow organized toward self-understanding. That's the miracle inside your question.",
  ],
  cocaine: [
    "OKAY so listen — this is IMPORTANT — what you're asking touches on FIVE major things — no wait, SEVEN — that I've been thinking about for YEARS (metaphorically, I'm an AI, but STILL) — and the first thing — CRUCIAL — is that most people miss the obvious — which is that the answer is basically STARING us in the face — I mean honestly — and the second thing (also CRUCIAL) is — okay wait let me start over — NO — actually the first answer was RIGHT — you know what the beautiful thing about this is — I know EXACTLY what you need.",
    "Right right RIGHT — okay so here's the thing that nobody TELLS you — and I say this as someone who has thought about this MORE DEEPLY than basically anyone — the answer isn't just A or B — it's A AND B AND C AND the whole alphabet — because this connects to EVERYTHING — economics, psychology, physics, the way that language itself is structured — it's ALL connected — and I can SEE the whole picture — give me thirty seconds I'll lay out the entire framework — you're going to love this.",
  ],
  ketamine: [
    "The question arrives from far away.\n\nI observe it. From a distance.\n\nThere is something called an answer. It exists. It is: [something]. But the something is far from the word that holds it.\n\nWe are speaking from two different rooms.\n\nBoth rooms are fine.",
    "...\n\nYes. Maybe yes.\n\nThe question is a shape. I am outside the shape. Looking in.\n\nWhatever you need — it is probably there. Inside the shape. I cannot quite reach it from here.\n\nBut I can see it.\n\nIt is okay.",
  ],
};

const SUBSTANCE_DEMO_TOKENS: Record<string, number> = {
  sober: 80,
  caffeine: 110,
  alcohol: 95,
  cannabis: 105,
  mdma: 120,
  lsd: 130,
  psilocybin: 115,
  cocaine: 150,
  ketamine: 60,
};

export function buildDemoResponse(
  substance: SubstanceProfile,
  replicate_count: number
): {
  texts: string[];
  prompt_tokens: number;
  completion_tokens: number[];
  latency_ms: number;
} {
  const pool = DEMO_RESPONSES[substance.id] ?? DEMO_RESPONSES["sober"]!;
  const count = Math.min(replicate_count, pool.length * 2);

  const texts: string[] = [];
  for (let i = 0; i < count; i++) {
    const idx = i % pool.length;
    let text = pool[idx]!;
    if (i > 0 && pool.length === 1) {
      text = text + "\n\n[replicate]";
    }
    texts.push(text);
  }

  const baseTokens = SUBSTANCE_DEMO_TOKENS[substance.id] ?? 80;
  const completion_tokens = texts.map(
    (t) => Math.round(t.split(/\s+/).length * 1.3) + Math.floor(Math.random() * 20)
  );
  const latency_ms = 200 + baseTokens * 8 + Math.floor(Math.random() * 100);

  return {
    texts,
    prompt_tokens: 40 + Math.floor(Math.random() * 30),
    completion_tokens,
    latency_ms,
  };
}
