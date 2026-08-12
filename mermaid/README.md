# Mermaid sources

Diagram sources for the figures that ship as hand-authored SVG in the book.

Most diagrams in [`../README.md`](../README.md) are still written inline as fenced
` ```mermaid ` blocks — GitHub renders those natively, so the fence *is* the source and there is
nothing to keep here. This folder holds the originals for the diagrams that were replaced by an
SVG in [`../images/`](../images/), so the Mermaid stays available to edit, re-render, or fall
back to.

## Why these were converted

Each of these had a structural idea that Mermaid's layout engine flattened — a metaphor, a
comparison, or a boundary. The rest are flowcharts and trees, which Mermaid draws correctly, so
converting them would have been change without gain.

| Source | Ships as | What Mermaid was flattening |
|---|---|---|
| `01-each-paradigm-shift-traded-control-for-capab.mmd` | `01-….svg` | a trade — capability up, determinism down |
| `03-agent-capability-ladder.mmd` | `03-….svg` | a ladder, drawn as a flat row |
| `11-the-three-rag-paradigms.mmd` | `11-….svg` | three paradigms to compare, stacked vertically |
| `13-two-stage-retrieval.mmd` | `13-….svg` | a funnel: 100 candidates down to 5 |
| `16-five-agent-loop-shapes.mmd` | `16-….svg` | five different topologies, squashed into one strip |
| `18-reasoning-strategies.mmd` | `18-….svg` | four reasoning shapes — the shapes are the point |
| `20-five-coordination-topologies.mmd` | `20-….svg` | where the arrows point is the whole argument |
| `22-12-factors-of-agentic-design.mmd` | `22-….svg` | a 3×4 grid plus twelve dependencies |
| `27-evaluation-pyramid.mmd` | `27-….svg` | a pyramid, drawn as five equal boxes |
| `30-the-three-serving-side-levers-and-which-late.mmd` | `30-….svg` | a mapping from lever to metric |
| `31-maestro-s-seven-layers.mmd` | `31-….svg` | seven layers in a narrow column |
| `36-mcp-vs-a2a.mmd` | `36-….svg` | two protocols whose *shapes* differ |
| `agent-loop.mmd` | `agent-loop.svg` | the loop, and the boundary running through it |

## Re-rendering a `.mmd`

The SVGs are hand-authored, not generated from these sources — they differ deliberately, since
each SVG carries layout the Mermaid could not express. To render the Mermaid itself:

```bash
npx -y @mermaid-js/mermaid-cli -i mermaid/27-evaluation-pyramid.mmd -o /tmp/out.svg
```

## Falling back to Mermaid

Replace the image line in `../README.md` with this file's contents wrapped in a ` ```mermaid `
fence. Nothing else depends on the SVG.
