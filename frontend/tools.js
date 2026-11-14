/**
 * Future-proof tool registry for the Neural Forge shell.
 * Add a tool later: one entry here + optional panel markup with data-tool-view.
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
      description: "Datasets, models, and predictions in your workspace.",
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
      description: "Keep preferred series and replay past what-ifs.",
      status: "coming-soon",
      view: "coming-soon",
    },
    {
      id: "what-if",
      name: "What-if",
      icon: "↝",
      description: "Counterfactual shocks on a forecast path.",
      status: "coming-soon",
      view: "coming-soon",
    },
    {
      id: "why",
      name: "Why this forecast",
      icon: "?",
      description: "Plain-language drivers behind a prediction.",
      status: "coming-soon",
      view: "coming-soon",
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
      id: "simulator",
      name: "Simulator",
      icon: "⟳",
      description: "Roll forward multi-step scenarios interactively.",
      status: "coming-soon",
      view: "coming-soon",
    },
  ];

  global.NeuralForgeTools = {
    TOOL_REGISTRY,
    liveTools() {
      return TOOL_REGISTRY.filter((tool) => tool.status === "live");
    },
    comingSoonTools() {
      return TOOL_REGISTRY.filter((tool) => tool.status === "coming-soon");
    },
    byId(id) {
      return TOOL_REGISTRY.find((tool) => tool.id === id) || null;
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
