/**
 * Future-proof tool registry for the Neural Forge shell.
 * Add a tool later: one entry here + optional panel markup with data-tool-view.
 * Landing "Tools & Roadmap" is driven by ROADMAP (statuses: live | in-progress | coming-soon).
 */
(function (global) {
  const TOOL_REGISTRY = [
    {
      id: "chat",
      name: "Chat",
      icon: "◇",
      description: "Plain-English agent with tools (GPT-OSS 120B).",
      status: "live",
      view: "chat",
    },
    {
      id: "forecast",
      name: "Forecast",
      icon: "↗",
      description: "Starter and workspace series forecasts with intervals.",
      status: "live",
      view: "forecast",
    },
    {
      id: "train",
      name: "Train on my CSV",
      icon: "▣",
      description: "Upload a CSV and train a private capped model.",
      status: "live",
      view: "train",
    },
    {
      id: "workspace",
      name: "My data / series",
      icon: "▤",
      description: "List, rename, delete, and retrain your uploaded CSV series.",
      status: "live",
      view: "workspace",
    },
    {
      id: "evaluation",
      name: "Accuracy / Evaluation",
      icon: "◎",
      description: "Backtested MAE, RMSE, and baseline comparison.",
      status: "live",
      view: "evaluation",
    },
    {
      id: "live",
      name: "Live data",
      icon: "◉",
      description: "Weather, markets, crypto, moon, and near-Earth space.",
      status: "live",
      view: "live",
    },
    {
      id: "docs",
      name: "API docs",
      icon: "{}",
      description: "HTTP examples and OpenAPI for developers.",
      status: "live",
      href: "/docs",
    },
    {
      id: "talk",
      name: "Talk (GQ voice)",
      icon: "♫",
      description: "Hands-free Talk mode and GQ voice credits.",
      status: "live",
      view: "voice",
    },
    {
      id: "memory",
      name: "Memory & saved scenarios",
      icon: "◈",
      description: "Preferences the chat remembers, plus named forecast scenarios you can re-run.",
      status: "live",
      view: "memory",
    },
    {
      id: "pca",
      name: "PCA / eigenvectors",
      icon: "⌖",
      description: "Find the main drivers in your data.",
      status: "live",
      view: "pca",
    },
    {
      id: "kmeans",
      name: "K-Means clustering",
      icon: "⊛",
      description: "Spot regimes and unusual points for alerts.",
      status: "live",
      view: "regimes",
    },
    {
      id: "diffusion",
      name: "Diffusion forecaster",
      icon: "≋",
      description: "A range of possible futures, not one path.",
      status: "live",
      view: "coming-soon",
    },
    {
      id: "jepa",
      name: "JEPA pretraining",
      icon: "⬡",
      description: "Smarter learned representations for series.",
      status: "live",
      view: "coming-soon",
    },
    {
      id: "what-if",
      name: "What-if",
      icon: "↝",
      description: "Counterfactual shocks on a forecast path.",
      status: "live",
      view: "what-if",
    },
    {
      id: "why",
      name: "Why this forecast",
      icon: "?",
      description: "Plain-language drivers behind a prediction.",
      status: "live",
      view: "why",
    },
    {
      id: "reports",
      name: "Reports",
      icon: "▥",
      description: "Shareable accuracy and forecast summaries.",
      status: "coming-soon",
      view: "coming-soon",
    },
    {
      id: "alerts",
      name: "Alerts",
      icon: "⚠",
      description: "Notify when a series crosses a threshold.",
      status: "coming-soon",
      view: "coming-soon",
    },
    {
      id: "synthetic",
      name: "Synthetic data",
      icon: "∿",
      description: "Generate training series with known patterns.",
      status: "coming-soon",
      view: "coming-soon",
    },
    {
      id: "multimodal",
      name: "Multimodal",
      icon: "⧉",
      description: "Charts, images, and text in one workflow.",
      status: "coming-soon",
      view: "coming-soon",
    },
    {
      id: "simulator",
      name: "Simulator",
      icon: "⟳",
      description: "Roll forward multi-step scenarios interactively.",
      status: "coming-soon",
      view: "coming-soon",
    },
  ];

  /** Landing-page phases. Update status/items here; sidebar uses TOOL_REGISTRY. */
  const ROADMAP = [
    {
      id: "live",
      title: "Already live",
      status: "live",
      items: [
        {
          name: "Grok-style dashboard",
          description: "A clean shell for chat, tools, and forecasts.",
        },
        {
          name: "Chat on GPT-OSS 120B",
          description: "Ask in plain English; the agent calls tools when needed.",
          toolId: "chat",
        },
        {
          name: "Talk mode",
          description: "Hands-free voice listen, answer, and speak.",
          toolId: "talk",
        },
        {
          name: "Train on my CSV",
          description: "Upload a series and train a private capped model.",
          toolId: "train",
        },
        {
          name: "Confidence & recent accuracy",
          description: "Every prediction shows intervals and how it has been doing.",
          toolId: "evaluation",
        },
        {
          name: "Live weather & prices",
          description: "Straight answers for weather, markets, and more.",
          toolId: "live",
        },
      ],
    },
    {
      id: "phase-2",
      title: "Phase 2 — Interactive & Stateful",
      status: "live",
      items: [
        {
          name: "Per-user memory",
          description: "Preferences the chat remembers across sessions.",
          toolId: "memory",
        },
        {
          name: "Saved chats in the sidebar",
          description: "Pick up where you left off from History.",
        },
        {
          name: "Saved scenarios",
          description: "Name forecast scenarios and re-run them later.",
          toolId: "memory",
        },
        {
          name: "Manage your own data series",
          description: "Rename, delete, and retrain uploaded CSV series.",
          toolId: "workspace",
        },
      ],
    },
    {
      id: "ml-tools",
      title: "New ML tools",
      status: "live",
      items: [
        {
          name: "PCA / eigenvectors",
          description: "Find the main drivers in your data.",
          toolId: "pca",
        },
        {
          name: "K-Means clustering",
          description: "Find weather or market regimes and flag unusual points.",
          toolId: "kmeans",
        },
        {
          name: "Diffusion forecaster",
          description: "See a range of possible futures, not one line.",
          toolId: "diffusion",
        },
        {
          name: "JEPA pretraining",
          description: "Smarter learned representations for your series.",
          toolId: "jepa",
        },
      ],
    },
    {
      id: "phase-3",
      title: "Phase 3 — Generative Layer",
      status: "in-progress",
      items: [
        {
          name: "Why this forecast",
          description: "Plain-language explanations behind a prediction.",
          toolId: "why",
        },
        {
          name: "What-if",
          description: "Shock a path and see how the forecast moves.",
          toolId: "what-if",
        },
        {
          name: "Reports",
          description: "Shareable accuracy and forecast summaries.",
          toolId: "reports",
        },
        {
          name: "Alerts",
          description: "Get notified when a series crosses a threshold.",
          toolId: "alerts",
        },
        {
          name: "Synthetic data",
          description: "Generate training series with known patterns.",
          toolId: "synthetic",
        },
      ],
    },
    {
      id: "phase-4",
      title: "Phase 4 — Multimodal",
      status: "coming-soon",
      items: [
        {
          name: "Multimodal inputs",
          description: "Work with charts, images, and text in one place.",
          toolId: "multimodal",
        },
      ],
    },
    {
      id: "phase-5",
      title: "Phase 5 — World Model / Simulator",
      status: "coming-soon",
      items: [
        {
          name: "Simulator",
          description: "Roll forward multi-step scenarios interactively.",
          toolId: "simulator",
        },
      ],
    },
  ];

  const STATUS_LABELS = {
    live: "Live",
    "in-progress": "In progress",
    "coming-soon": "Coming soon",
  };

  function statusLabel(status) {
    return STATUS_LABELS[status] || status;
  }

  global.NeuralForgeTools = {
    TOOL_REGISTRY,
    ROADMAP,
    STATUS_LABELS,
    statusLabel,
    liveTools() {
      return TOOL_REGISTRY.filter((tool) => tool.status === "live");
    },
    inProgressTools() {
      return TOOL_REGISTRY.filter((tool) => tool.status === "in-progress");
    },
    comingSoonTools() {
      return TOOL_REGISTRY.filter((tool) => tool.status === "coming-soon");
    },
    byId(id) {
      return TOOL_REGISTRY.find((tool) => tool.id === id) || null;
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
