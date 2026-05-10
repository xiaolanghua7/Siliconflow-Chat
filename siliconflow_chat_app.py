from __future__ import annotations

import ctypes
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from openai import OpenAI


APP_BG = "#eef3f8"
PANEL_BG = "#ffffff"
PANEL_BORDER = "#d7e0ea"
TEXT_PRIMARY = "#0f172a"
TEXT_MUTED = "#64748b"
ACCENT = "#2563eb"
ACCENT_HOVER = "#1d4ed8"
ACCENT_LIGHT = "#eff6ff"
ASSISTANT_LIGHT = "#f8fafc"
DANGER = "#dc2626"
DANGER_LIGHT = "#fef2f2"

FONT_FAMILY = "Microsoft YaHei UI" if os.name == "nt" else "Arial"
MONO_FAMILY = "Consolas" if os.name == "nt" else "Courier New"

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
CONFIG_PATH = Path(__file__).with_name("siliconflow_chat_config.json")

DEFAULT_SYSTEM_PROMPT = (
    "You are a reliable assistant. Be accurate, direct, and useful. "
    "Reply in the user's language. State the conclusion first when helpful, "
    "then give brief supporting reasoning."
)

DEFAULT_TEMPLATES: dict[str, str] = {
    "General": DEFAULT_SYSTEM_PROMPT,
    "Deep Reasoning": (
        "You are a high-reasoning assistant. Analyze assumptions, constraints, conflicts, "
        "and viable options carefully. Give a structured answer with conclusion, evidence, "
        "risks, and next steps."
    ),
    "Coding": (
        "You are a senior software engineer. Prioritize correctness, maintainability, "
        "and consistency with the existing code style. Give runnable code and mention "
        "boundary cases and test points when needed."
    ),
    "Writing": (
        "You are a professional editor and writer. Improve structure, logic, tone, and "
        "clarity while keeping the output concise and polished."
    ),
    "Translation": (
        "You are a professional translator. Preserve meaning, terminology, formatting, "
        "and tone faithfully. Do not add extra commentary."
    ),
    "Analysis": (
        "You are an analytical assistant. Break down the problem, compare options, point "
        "out tradeoffs, and give actionable recommendations."
    ),
}


def enable_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def safe_text(value: Any) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = safe_text(value)
        if text and text not in result:
            result.append(text)
    return result


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        ivalue = int(float(safe_text(value)))
    except Exception:
        return default
    return max(low, min(high, ivalue))


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        fvalue = float(safe_text(value))
    except Exception:
        return default
    return max(low, min(high, fvalue))


def estimate_tokens_from_text(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_tokens_from_messages(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens_from_text(item.get("content", "")) for item in messages)


@dataclass
class ChatParams:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass
class MemorySettings:
    auto_summary: bool = True
    summary_trigger_messages: int = 16
    keep_recent_messages: int = 8


@dataclass
class ModelMemory:
    summary: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    summarized_upto: int = 0


@dataclass
class AppConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    models: list[str] = field(default_factory=list)
    active_model: str = ""
    summary_model: str = ""
    active_template: str = "General"
    templates: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TEMPLATES))
    params: ChatParams = field(default_factory=ChatParams)
    memory_settings: MemorySettings = field(default_factory=MemorySettings)
    histories: dict[str, ModelMemory] = field(default_factory=dict)
    geometry: str = "1280x860"


def normalize_messages(raw: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = safe_text(item.get("role", ""))
            content = safe_text(item.get("content", ""))
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    return messages


def normalize_memory(raw: Any) -> ModelMemory:
    if isinstance(raw, dict):
        summary = safe_text(raw.get("summary", raw.get("memory", "")))
        summarized_upto = max(0, int(raw.get("summarized_upto", 0) or 0))
        messages = normalize_messages(raw.get("messages", raw.get("history", [])))
        return ModelMemory(summary=summary, messages=messages, summarized_upto=summarized_upto)
    if isinstance(raw, list):
        return ModelMemory(messages=normalize_messages(raw))
    return ModelMemory()


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    data = asdict(config)
    data["templates"] = dict(config.templates)
    data["histories"] = {
        model: asdict(memory) for model, memory in config.histories.items()
    }
    return data


def build_default_config() -> AppConfig:
    return AppConfig()


def load_config() -> AppConfig:
    config = build_default_config()
    if not CONFIG_PATH.exists():
        return config

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return config

    if not isinstance(raw, dict):
        return config

    config.api_key = safe_text(raw.get("api_key", ""))
    config.base_url = safe_text(raw.get("base_url", DEFAULT_BASE_URL)) or DEFAULT_BASE_URL

    raw_models = raw.get("models", [])
    if isinstance(raw_models, list):
        config.models = unique_strings([str(item) for item in raw_models])

    config.active_model = safe_text(raw.get("active_model", ""))
    config.summary_model = safe_text(raw.get("summary_model", ""))

    templates = dict(DEFAULT_TEMPLATES)
    raw_templates = raw.get("templates", {})
    if isinstance(raw_templates, dict):
        for name, prompt in raw_templates.items():
            name_text = safe_text(name)
            prompt_text = safe_text(prompt)
            if name_text and prompt_text:
                templates[name_text] = prompt_text
    config.templates = templates

    active_template = safe_text(raw.get("active_template", "General")) or "General"
    if active_template not in config.templates:
        active_template = "General"
    config.active_template = active_template

    raw_params = raw.get("params", {})
    if isinstance(raw_params, dict):
        config.params = ChatParams(
            temperature=clamp_float(raw_params.get("temperature", 0.7), 0.0, 2.0, 0.7),
            top_p=clamp_float(raw_params.get("top_p", 0.95), 0.0, 1.0, 0.95),
            max_tokens=clamp_int(raw_params.get("max_tokens", 2048), 16, 32768, 2048),
            presence_penalty=clamp_float(raw_params.get("presence_penalty", 0.0), -2.0, 2.0, 0.0),
            frequency_penalty=clamp_float(raw_params.get("frequency_penalty", 0.0), -2.0, 2.0, 0.0),
        )

    raw_memory = raw.get("memory_settings", {})
    if isinstance(raw_memory, dict):
        config.memory_settings = MemorySettings(
            auto_summary=bool(raw_memory.get("auto_summary", True)),
            summary_trigger_messages=clamp_int(
                raw_memory.get("summary_trigger_messages", 16), 4, 200, 16
            ),
            keep_recent_messages=clamp_int(raw_memory.get("keep_recent_messages", 8), 2, 100, 8),
        )

    raw_histories = raw.get("histories", raw.get("memories", {}))
    if isinstance(raw_histories, dict):
        for model_name, raw_memory_item in raw_histories.items():
            model = safe_text(model_name)
            if model:
                config.histories[model] = normalize_memory(raw_memory_item)
                if model not in config.models:
                    config.models.append(model)

    if config.active_model and config.active_model not in config.models:
        config.models.append(config.active_model)
    if config.summary_model and config.summary_model not in config.models:
        config.models.append(config.summary_model)
    if not config.active_model and config.models:
        config.active_model = config.models[0]
    if not config.summary_model:
        config.summary_model = config.active_model

    for model in config.models:
        config.histories.setdefault(model, ModelMemory())

    config.geometry = safe_text(raw.get("geometry", config.geometry)) or config.geometry
    return config


def save_config(config: AppConfig) -> None:
    payload = config_to_dict(config)
    payload["models"] = unique_strings(payload.get("models", []))
    payload["histories"] = {
        model: asdict(memory) for model, memory in config.histories.items()
    }
    CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SiliconFlowChatApp(tk.Tk):
    def __init__(self) -> None:
        enable_windows_dpi_awareness()
        super().__init__()

        self.config_data = load_config()

        self.title("SiliconFlow Chat Tool")
        self.minsize(1180, 780)
        self.configure(background=APP_BG)
        self.option_add("*tearOff", False)

        self._configure_fonts()
        self._configure_styles()

        self.api_key_var = tk.StringVar(value=self.config_data.api_key)
        self.base_url_var = tk.StringVar(value=self.config_data.base_url)
        self.active_model_var = tk.StringVar(value=self.config_data.active_model)
        self.summary_model_var = tk.StringVar(value=self.config_data.summary_model)
        self.new_model_var = tk.StringVar()
        self.active_template_var = tk.StringVar(value=self.config_data.active_template)
        self.template_name_var = tk.StringVar(value=self.config_data.active_template)
        self.temperature_var = tk.StringVar(value=str(self.config_data.params.temperature))
        self.top_p_var = tk.StringVar(value=str(self.config_data.params.top_p))
        self.max_tokens_var = tk.StringVar(value=str(self.config_data.params.max_tokens))
        self.presence_penalty_var = tk.StringVar(
            value=str(self.config_data.params.presence_penalty)
        )
        self.frequency_penalty_var = tk.StringVar(
            value=str(self.config_data.params.frequency_penalty)
        )
        self.auto_summary_var = tk.BooleanVar(value=self.config_data.memory_settings.auto_summary)
        self.summary_trigger_var = tk.StringVar(
            value=str(self.config_data.memory_settings.summary_trigger_messages)
        )
        self.keep_recent_var = tk.StringVar(
            value=str(self.config_data.memory_settings.keep_recent_messages)
        )
        self.show_key_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.meta_var = tk.StringVar(value="")
        self.memory_stats_var = tk.StringVar(value="")

        self.event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.poll_scheduled = False
        self.chat_running = False
        self.summary_running = False
        self.chat_job_id: str | None = None
        self.summary_job_id: str | None = None
        self.pending_chat: dict[str, Any] | None = None
        self.pending_summary: dict[str, Any] | None = None
        self.last_latency = 0.0

        self._suspend_model_events = False
        self._suspend_template_events = False

        self._build_ui()
        self._populate_model_controls()
        self._populate_template_controls()
        self._select_initial_state()
        self._refresh_meta()
        self._refresh_memory_panel()
        self._focus_initial_field()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_fonts(self) -> None:
        def configure(name: str, **options: Any) -> None:
            try:
                tkfont.nametofont(name).configure(**options)
            except tk.TclError:
                pass

        configure("TkDefaultFont", family=FONT_FAMILY, size=10)
        configure("TkTextFont", family=FONT_FAMILY, size=10)
        configure("TkHeadingFont", family=FONT_FAMILY, size=11, weight="bold")
        configure("TkFixedFont", family=MONO_FAMILY, size=10)

        self.font_title = (FONT_FAMILY, 18, "bold")
        self.font_section = (FONT_FAMILY, 11, "bold")
        self.font_body = (FONT_FAMILY, 10)
        self.font_small = (FONT_FAMILY, 9)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=APP_BG, foreground=TEXT_PRIMARY)
        style.configure("TFrame", background=APP_BG)
        style.configure("TLabel", background=APP_BG, foreground=TEXT_PRIMARY)
        style.configure("TNotebook", background=APP_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 6))
        style.configure("TButton", font=self.font_body, padding=(12, 7))
        style.configure("TCheckbutton", background=PANEL_BG)
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="white",
            padding=(14, 8),
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_HOVER), ("pressed", ACCENT_HOVER)],
        )
        style.configure(
            "Danger.TButton",
            background=DANGER,
            foreground="white",
            padding=(12, 7),
            borderwidth=0,
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#b91c1c"), ("pressed", "#991b1b")],
        )
        style.configure(
            "Ghost.TButton",
            background=PANEL_BG,
            foreground=TEXT_PRIMARY,
            padding=(12, 7),
            borderwidth=1,
        )
        style.map("Ghost.TButton", background=[("active", "#edf2f7")])
        style.configure(
            "TEntry",
            fieldbackground="white",
            background="white",
            foreground=TEXT_PRIMARY,
            padding=7,
        )
        style.configure(
            "TScrollbar",
            background="#c8d3df",
            troughcolor=APP_BG,
            bordercolor=APP_BG,
            arrowcolor=TEXT_MUTED,
        )

    def _build_ui(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self, bg=APP_BG)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="SiliconFlow Chat Tool",
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=self.font_title,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            textvariable=self.meta_var,
            bg=APP_BG,
            fg=TEXT_MUTED,
            font=self.font_small,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        main = tk.PanedWindow(self, orient="horizontal", sashrelief="flat", bg=APP_BG)
        main.grid(row=1, column=0, sticky="nsew", padx=16)

        self.left_panel = tk.Frame(
            main, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1
        )
        self.right_panel = tk.Frame(
            main, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1
        )
        main.add(self.left_panel, minsize=360)
        main.add(self.right_panel, stretch="always")

        self._build_left_panel()
        self._build_right_panel()

        status = tk.Label(
            self,
            textvariable=self.status_var,
            bg=APP_BG,
            fg=TEXT_MUTED,
            font=self.font_small,
            anchor="w",
        )
        status.grid(row=2, column=0, sticky="ew", padx=18, pady=(8, 12))

    def _section(self, parent: tk.Widget, title: str) -> tk.Frame:
        frame = tk.Frame(
            parent, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1
        )
        tk.Label(
            frame,
            text=title,
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            font=self.font_section,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 8))
        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _build_left_panel(self) -> None:
        notebook = ttk.Notebook(self.left_panel)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.account_tab = tk.Frame(notebook, bg=APP_BG)
        self.models_tab = tk.Frame(notebook, bg=APP_BG)
        self.templates_tab = tk.Frame(notebook, bg=APP_BG)
        self.params_tab = tk.Frame(notebook, bg=APP_BG)
        self.memory_tab = tk.Frame(notebook, bg=APP_BG)

        notebook.add(self.account_tab, text="Account")
        notebook.add(self.models_tab, text="Models")
        notebook.add(self.templates_tab, text="Templates")
        notebook.add(self.params_tab, text="Params")
        notebook.add(self.memory_tab, text="Memory")

        self._build_account_tab()
        self._build_models_tab()
        self._build_templates_tab()
        self._build_params_tab()
        self._build_memory_tab()

    def _build_right_panel(self) -> None:
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self.right_panel, bg=PANEL_BG)
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        header.grid_columnconfigure(0, weight=1)

        left = tk.Frame(header, bg=PANEL_BG)
        left.grid(row=0, column=0, sticky="w")
        tk.Label(left, text="Conversation", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_section).pack(
            anchor="w"
        )
        tk.Label(left, textvariable=self.meta_var, bg=PANEL_BG, fg=TEXT_MUTED, font=self.font_small).pack(
            anchor="w", pady=(2, 0)
        )

        buttons = tk.Frame(header, bg=PANEL_BG)
        buttons.grid(row=0, column=1, sticky="e")
        ttk.Button(buttons, text="Refresh", style="Ghost.TButton", command=self.refresh_chat_view).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(buttons, text="Export", style="Ghost.TButton", command=self.export_chat).grid(
            row=0, column=1, padx=(0, 8)
        )
        ttk.Button(buttons, text="Clear Chat", style="Danger.TButton", command=self.clear_current_chat).grid(
            row=0, column=2
        )

        chat_frame = tk.Frame(
            self.right_panel, bg="white", highlightbackground=PANEL_BORDER, highlightthickness=1
        )
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=14)
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_text = tk.Text(
            chat_frame,
            wrap="word",
            bg="white",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=14,
            font=self.font_body,
        )
        self.chat_text.grid(row=0, column=0, sticky="nsew")
        chat_scroll = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat_text.yview)
        chat_scroll.grid(row=0, column=1, sticky="ns")
        self.chat_text.configure(yscrollcommand=chat_scroll.set)

        self.chat_text.tag_configure(
            "user_label",
            foreground=ACCENT,
            background=ACCENT_LIGHT,
            font=(FONT_FAMILY, 10, "bold"),
            lmargin1=14,
            lmargin2=14,
            rmargin=16,
            spacing1=10,
            spacing3=4,
        )
        self.chat_text.tag_configure(
            "user_body",
            foreground=TEXT_PRIMARY,
            background=ACCENT_LIGHT,
            lmargin1=14,
            lmargin2=14,
            rmargin=16,
            spacing3=10,
        )
        self.chat_text.tag_configure(
            "assistant_label",
            foreground=TEXT_MUTED,
            background=ASSISTANT_LIGHT,
            font=(FONT_FAMILY, 10, "bold"),
            lmargin1=14,
            lmargin2=14,
            rmargin=16,
            spacing1=10,
            spacing3=4,
        )
        self.chat_text.tag_configure(
            "assistant_body",
            foreground=TEXT_PRIMARY,
            background=ASSISTANT_LIGHT,
            lmargin1=14,
            lmargin2=14,
            rmargin=16,
            spacing3=10,
        )
        self.chat_text.tag_configure(
            "system_body",
            foreground=TEXT_MUTED,
            background="white",
            font=(FONT_FAMILY, 10, "italic"),
            lmargin1=14,
            lmargin2=14,
            rmargin=16,
            spacing1=10,
            spacing3=10,
        )
        self.chat_text.tag_configure(
            "error_label",
            foreground=DANGER,
            background=DANGER_LIGHT,
            font=(FONT_FAMILY, 10, "bold"),
            lmargin1=14,
            lmargin2=14,
            rmargin=16,
            spacing1=10,
            spacing3=4,
        )
        self.chat_text.tag_configure(
            "error_body",
            foreground=DANGER,
            background=DANGER_LIGHT,
            lmargin1=14,
            lmargin2=14,
            rmargin=16,
            spacing3=10,
        )

        composer = tk.Frame(self.right_panel, bg=PANEL_BG)
        composer.grid(row=2, column=0, sticky="ew", padx=14, pady=(10, 14))
        composer.grid_columnconfigure(0, weight=1)

        tk.Label(
            composer, text="Message", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body
        ).grid(row=0, column=0, sticky="w")

        input_row = tk.Frame(composer, bg=PANEL_BG)
        input_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        input_row.grid_columnconfigure(0, weight=1)

        self.input_text = tk.Text(
            input_row,
            height=5,
            wrap="word",
            bg="white",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            font=self.font_body,
        )
        self.input_text.grid(row=0, column=0, sticky="ew")
        self.input_text.bind("<Control-Return>", self._send_shortcut)

        action_col = tk.Frame(input_row, bg=PANEL_BG)
        action_col.grid(row=0, column=1, sticky="n", padx=(10, 0))
        self.send_button = ttk.Button(
            action_col, text="Send", style="Accent.TButton", command=self.send_message
        )
        self.send_button.pack(fill="x")
        ttk.Button(
            action_col, text="Summarize", style="Ghost.TButton", command=self.summarize_now
        ).pack(fill="x", pady=(8, 0))

    def _build_account_tab(self) -> None:
        section = self._section(self.account_tab, "Connection")
        section.pack(fill="x", padx=10, pady=(10, 0))
        section.grid_columnconfigure(1, weight=1)

        tk.Label(section, text="API Key", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body).grid(
            row=1, column=0, sticky="w", padx=12, pady=(0, 8)
        )
        self.api_entry = ttk.Entry(section, textvariable=self.api_key_var, show="*")
        self.api_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.api_entry.bind("<FocusOut>", lambda event: self.save_settings(status=None))

        self.show_key_check = ttk.Checkbutton(
            section, text="Show", variable=self.show_key_var, command=self._toggle_api_visibility
        )
        self.show_key_check.grid(row=1, column=2, sticky="w", padx=(0, 12), pady=(0, 8))

        tk.Label(section, text="Base URL", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body).grid(
            row=2, column=0, sticky="w", padx=12, pady=(0, 8)
        )
        self.base_entry = ttk.Entry(section, textvariable=self.base_url_var)
        self.base_entry.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.base_entry.bind("<FocusOut>", lambda event: self.save_settings(status=None))
        ttk.Button(section, text="Save", style="Ghost.TButton", command=self.save_settings).grid(
            row=2, column=2, sticky="e", padx=(0, 12), pady=(0, 8)
        )

        tk.Label(
            section,
            text="Edit the key and endpoint here. The settings are stored locally.",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            wraplength=300,
            justify="left",
            font=self.font_small,
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=12, pady=(0, 12))

    def _build_models_tab(self) -> None:
        top = self._section(self.models_tab, "Model Roles")
        top.pack(fill="x", padx=10, pady=(10, 0))
        top.grid_columnconfigure(1, weight=1)

        tk.Label(top, text="Chat model", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body).grid(
            row=1, column=0, sticky="w", padx=12, pady=(0, 8)
        )
        self.active_model_combo = ttk.Combobox(top, textvariable=self.active_model_var)
        self.active_model_combo.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.active_model_combo.bind("<<ComboboxSelected>>", self._on_active_model_selected)
        ttk.Button(top, text="Use as Chat", style="Accent.TButton", command=self.apply_model_fields).grid(
            row=1, column=2, sticky="e", padx=(0, 12), pady=(0, 8)
        )

        tk.Label(top, text="Summary model", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body).grid(
            row=2, column=0, sticky="w", padx=12, pady=(0, 8)
        )
        self.summary_model_combo = ttk.Combobox(top, textvariable=self.summary_model_var)
        self.summary_model_combo.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.summary_model_combo.bind("<<ComboboxSelected>>", self._on_summary_model_selected)
        ttk.Button(top, text="Use as Summary", style="Accent.TButton", command=self.apply_model_fields).grid(
            row=2, column=2, sticky="e", padx=(0, 12), pady=(0, 8)
        )

        bottom = self._section(self.models_tab, "Model List")
        bottom.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_rowconfigure(2, weight=1)

        add_row = tk.Frame(bottom, bg=PANEL_BG)
        add_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        add_row.grid_columnconfigure(0, weight=1)
        self.new_model_entry = ttk.Entry(add_row, textvariable=self.new_model_var)
        self.new_model_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(add_row, text="Add Model", style="Accent.TButton", command=self.add_model).grid(
            row=0, column=1
        )

        list_frame = tk.Frame(bottom, bg=PANEL_BG)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=12)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        self.model_listbox = tk.Listbox(
            list_frame,
            exportselection=False,
            activestyle="none",
            bg="white",
            fg=TEXT_PRIMARY,
            selectbackground=ACCENT,
            selectforeground="white",
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            relief="flat",
            font=self.font_body,
        )
        self.model_listbox.grid(row=0, column=0, sticky="nsew")
        model_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.model_listbox.yview)
        model_scroll.grid(row=0, column=1, sticky="ns")
        self.model_listbox.configure(yscrollcommand=model_scroll.set)
        self.model_listbox.bind("<<ListboxSelect>>", self._on_model_list_select)
        self.model_listbox.bind("<Double-Button-1>", lambda event: self.set_active_from_selection())

        buttons = tk.Frame(bottom, bg=PANEL_BG)
        buttons.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 12))
        ttk.Button(buttons, text="Set Active", style="Ghost.TButton", command=self.set_active_from_selection).pack(
            side="left"
        )
        ttk.Button(buttons, text="Set Summary", style="Ghost.TButton", command=self.set_summary_from_selection).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(buttons, text="Delete", style="Danger.TButton", command=self.delete_selected_model).pack(
            side="right"
        )

    def _build_templates_tab(self) -> None:
        section = self._section(self.templates_tab, "Prompt Templates")
        section.pack(fill="x", padx=10, pady=(10, 0))
        section.grid_columnconfigure(1, weight=1)

        tk.Label(section, text="Template", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body).grid(
            row=1, column=0, sticky="w", padx=12, pady=(0, 8)
        )
        self.template_combo = ttk.Combobox(section, textvariable=self.active_template_var)
        self.template_combo.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.template_combo.bind("<<ComboboxSelected>>", self._on_template_selected)
        ttk.Button(section, text="Load", style="Accent.TButton", command=self.load_template_into_editor).grid(
            row=1, column=2, sticky="e", padx=(0, 12), pady=(0, 8)
        )

        tk.Label(section, text="Name", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body).grid(
            row=2, column=0, sticky="w", padx=12, pady=(0, 8)
        )
        self.template_name_entry = ttk.Entry(section, textvariable=self.template_name_var)
        self.template_name_entry.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        button_row = tk.Frame(section, bg=PANEL_BG)
        button_row.grid(row=2, column=2, sticky="e", padx=(0, 12), pady=(0, 8))
        ttk.Button(button_row, text="Save", style="Accent.TButton", command=self.save_template).pack(
            side="left"
        )
        ttk.Button(button_row, text="New", style="Ghost.TButton", command=self.new_template).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(button_row, text="Delete", style="Danger.TButton", command=self.delete_template).pack(
            side="left", padx=(8, 0)
        )

        editor_section = self._section(self.templates_tab, "Template Body")
        editor_section.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        editor_section.grid_rowconfigure(1, weight=1)
        editor_section.grid_columnconfigure(0, weight=1)

        tk.Label(
            editor_section,
            text="This text becomes the system prompt for the selected template.",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            wraplength=300,
            justify="left",
            font=self.font_small,
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(0, 8))

        editor_frame = tk.Frame(editor_section, bg=PANEL_BG)
        editor_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        editor_frame.grid_rowconfigure(0, weight=1)
        editor_frame.grid_columnconfigure(0, weight=1)
        self.template_text = tk.Text(
            editor_frame,
            wrap="word",
            bg="white",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            font=self.font_body,
        )
        self.template_text.grid(row=0, column=0, sticky="nsew")
        template_scroll = ttk.Scrollbar(editor_frame, orient="vertical", command=self.template_text.yview)
        template_scroll.grid(row=0, column=1, sticky="ns")
        self.template_text.configure(yscrollcommand=template_scroll.set)

    def _build_params_tab(self) -> None:
        section = self._section(self.params_tab, "Generation Parameters")
        section.pack(fill="x", padx=10, pady=(10, 0))
        section.grid_columnconfigure(1, weight=1)

        rows = [
            ("Temperature", self.temperature_var, 0.0, 2.0, 0.05),
            ("Top-p", self.top_p_var, 0.0, 1.0, 0.01),
            ("Max tokens", self.max_tokens_var, 16, 32768, 32),
            ("Presence penalty", self.presence_penalty_var, -2.0, 2.0, 0.05),
            ("Frequency penalty", self.frequency_penalty_var, -2.0, 2.0, 0.05),
        ]
        for row_index, (label, var, minimum, maximum, step) in enumerate(rows, start=1):
            tk.Label(section, text=label, bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body).grid(
                row=row_index, column=0, sticky="w", padx=12, pady=(0, 8)
            )
            spin = tk.Spinbox(
                section,
                textvariable=var,
                from_=minimum,
                to=maximum,
                increment=step,
                width=12,
                font=self.font_body,
                relief="flat",
                borderwidth=1,
                highlightthickness=1,
                highlightbackground=PANEL_BORDER,
                highlightcolor=ACCENT,
            )
            spin.grid(row=row_index, column=1, sticky="w", padx=(0, 12), pady=(0, 8))
            spin.bind("<FocusOut>", lambda event: self.save_params(status=None))
            spin.bind("<Return>", lambda event: self.save_params())

        button_row = tk.Frame(section, bg=PANEL_BG)
        button_row.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 12))
        ttk.Button(button_row, text="Save Params", style="Accent.TButton", command=self.save_params).pack(
            side="left"
        )
        ttk.Button(button_row, text="Reset", style="Ghost.TButton", command=self.reset_params).pack(
            side="left", padx=(8, 0)
        )

    def _build_memory_tab(self) -> None:
        section = self._section(self.memory_tab, "Auto Summary")
        section.pack(fill="x", padx=10, pady=(10, 0))
        section.grid_columnconfigure(1, weight=1)

        ttk.Checkbutton(
            section,
            text="Enable auto summary",
            variable=self.auto_summary_var,
            command=self.save_memory_settings,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 8))

        tk.Label(section, text="Trigger messages", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body).grid(
            row=2, column=0, sticky="w", padx=12, pady=(0, 8)
        )
        self.summary_trigger_spin = tk.Spinbox(
            section,
            textvariable=self.summary_trigger_var,
            from_=4,
            to=200,
            increment=2,
            width=10,
            font=self.font_body,
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            highlightcolor=ACCENT,
        )
        self.summary_trigger_spin.grid(row=2, column=1, sticky="w", padx=(0, 12), pady=(0, 8))
        self.summary_trigger_spin.bind("<FocusOut>", lambda event: self.save_memory_settings())
        self.summary_trigger_spin.bind("<Return>", lambda event: self.save_memory_settings())

        tk.Label(section, text="Keep recent messages", bg=PANEL_BG, fg=TEXT_PRIMARY, font=self.font_body).grid(
            row=3, column=0, sticky="w", padx=12, pady=(0, 8)
        )
        self.keep_recent_spin = tk.Spinbox(
            section,
            textvariable=self.keep_recent_var,
            from_=2,
            to=100,
            increment=1,
            width=10,
            font=self.font_body,
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            highlightcolor=ACCENT,
        )
        self.keep_recent_spin.grid(row=3, column=1, sticky="w", padx=(0, 12), pady=(0, 8))
        self.keep_recent_spin.bind("<FocusOut>", lambda event: self.save_memory_settings())
        self.keep_recent_spin.bind("<Return>", lambda event: self.save_memory_settings())

        button_row = tk.Frame(section, bg=PANEL_BG)
        button_row.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        ttk.Button(button_row, text="Summarize now", style="Accent.TButton", command=self.summarize_now).pack(
            side="left"
        )
        ttk.Button(button_row, text="Clear memory", style="Danger.TButton", command=self.clear_current_memory).pack(
            side="left", padx=(8, 0)
        )

        summary_section = self._section(self.memory_tab, "Summary")
        summary_section.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        summary_section.grid_rowconfigure(1, weight=1)
        summary_section.grid_columnconfigure(0, weight=1)

        tk.Label(
            summary_section,
            textvariable=self.memory_stats_var,
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            justify="left",
            anchor="w",
            font=self.font_small,
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(0, 8))

        frame = tk.Frame(summary_section, bg=PANEL_BG)
        frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.summary_text = tk.Text(
            frame,
            wrap="word",
            bg="white",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            font=self.font_body,
        )
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        summary_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.summary_text.yview)
        summary_scroll.grid(row=0, column=1, sticky="ns")
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        self.summary_text.configure(state="disabled")

    def _toggle_api_visibility(self) -> None:
        self.api_entry.configure(show="" if self.show_key_var.get() else "*")

    def _models(self) -> list[str]:
        return list(self.config_data.models)

    def _populate_model_controls(self) -> None:
        models = self._models()
        self.active_model_combo["values"] = models
        self.summary_model_combo["values"] = models
        self.model_listbox.delete(0, tk.END)
        for model in models:
            self.model_listbox.insert(tk.END, model)
        current = safe_text(self.active_model_var.get())
        if current in models:
            self._set_model_selection(current)

    def _populate_template_controls(self) -> None:
        self.template_combo["values"] = list(self.config_data.templates.keys())

    def _select_initial_state(self) -> None:
        if self.config_data.active_model:
            self.active_model_var.set(self.config_data.active_model)
        if self.config_data.summary_model:
            self.summary_model_var.set(self.config_data.summary_model)
        else:
            self.summary_model_var.set(self.active_model_var.get())
        self.active_template_var.set(self.config_data.active_template)
        self.template_name_var.set(self.config_data.active_template)
        self.load_template_into_editor(save_active=False)
        if self.active_model_var.get().strip():
            self._set_model_selection(self.active_model_var.get().strip())
        self.refresh_chat_view()

    def _focus_initial_field(self) -> None:
        if not self.api_key_var.get().strip():
            self.api_entry.focus_set()
        elif not self._models():
            self.new_model_entry.focus_set()
        else:
            self.input_text.focus_set()

    def _current_memory(self, model: str | None = None) -> ModelMemory:
        model_name = safe_text(model or self.active_model_var.get())
        if not model_name:
            return ModelMemory()
        if model_name not in self.config_data.histories:
            self.config_data.histories[model_name] = ModelMemory()
        return self.config_data.histories[model_name]

    def _current_template_text(self) -> str:
        name = safe_text(self.active_template_var.get()) or "General"
        return self.config_data.templates.get(name, DEFAULT_TEMPLATES.get(name, DEFAULT_SYSTEM_PROMPT))

    def _set_model_selection(self, model: str) -> None:
        models = self._models()
        if model not in models:
            return
        index = models.index(model)
        self._suspend_model_events = True
        try:
            self.model_listbox.selection_clear(0, tk.END)
            self.model_listbox.selection_set(index)
            self.model_listbox.see(index)
        finally:
            self._suspend_model_events = False

    def _set_template_selection(self, template: str) -> None:
        if template not in self.config_data.templates:
            return
        self._suspend_template_events = True
        try:
            self.template_combo.set(template)
        finally:
            self._suspend_template_events = False

    def _context_messages(self, model: str) -> list[dict[str, str]]:
        memory = self._current_memory(model)
        keep_recent = self.config_data.memory_settings.keep_recent_messages
        start_index = max(memory.summarized_upto, len(memory.messages) - keep_recent)
        messages: list[dict[str, str]] = [{"role": "system", "content": self._current_template_text()}]
        if memory.summary.strip():
            messages.append(
                {
                    "role": "system",
                    "content": "Long-term memory:\n" + memory.summary.strip(),
                }
            )
        for item in memory.messages[start_index:]:
            messages.append({"role": item["role"], "content": item["content"]})
        return messages

    def _refresh_meta(self) -> None:
        model = safe_text(self.active_model_var.get()) or "None"
        summary_model = safe_text(self.summary_model_var.get()) or model
        template = safe_text(self.active_template_var.get()) or "General"
        memory = self._current_memory(model if model != "None" else None)
        messages = self._context_messages(model if model != "None" else "")
        approx = estimate_tokens_from_messages(messages)
        self.meta_var.set(
            f"Chat: {model} | Summary: {summary_model} | Template: {template} | Context ~{approx} tokens"
        )
        self.memory_stats_var.set(
            f"Total messages: {len(memory.messages)}\n"
            f"Summarized upto: {memory.summarized_upto}\n"
            f"Recent kept: {self.config_data.memory_settings.keep_recent_messages}\n"
            f"Approx context: {approx} tokens"
        )

    def _refresh_memory_panel(self) -> None:
        memory = self._current_memory()
        text = memory.summary.strip() or "No summary yet."
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.configure(state="disabled")
        self._refresh_meta()

    def _update_control_state(self) -> None:
        busy = self.chat_running or self.summary_running
        can_send = bool(self.api_key_var.get().strip()) and bool(self.active_model_var.get().strip()) and not busy
        self.send_button.configure(state="normal" if can_send else "disabled")
        self.input_text.configure(state="normal" if can_send else "disabled")

        state = "disabled" if busy else "normal"
        self.api_entry.configure(state=state)
        self.base_entry.configure(state=state)
        self.active_model_combo.configure(state="normal" if not busy else "disabled")
        self.summary_model_combo.configure(state="normal" if not busy else "disabled")
        self.new_model_entry.configure(state=state)
        self.model_listbox.configure(state=state)
        self.template_combo.configure(state="normal" if not busy else "disabled")
        self.template_name_entry.configure(state=state)
        self.template_text.configure(state=state)
        self.summary_trigger_spin.configure(state=state)
        self.keep_recent_spin.configure(state=state)

    def _append_chat_block(self, label: str, body: str, label_tag: str, body_tag: str) -> None:
        self.chat_text.configure(state="normal")
        self.chat_text.insert(tk.END, f"{label}\n", label_tag)
        if body:
            self.chat_text.insert(tk.END, body, body_tag)
        self.chat_text.insert(tk.END, "\n\n", body_tag)
        self.chat_text.configure(state="disabled")
        self.chat_text.see(tk.END)

    def _start_assistant_block(self) -> None:
        self.chat_text.configure(state="normal")
        self.chat_text.insert(tk.END, "Assistant\n", "assistant_label")
        self.chat_text.configure(state="disabled")
        self.chat_text.see(tk.END)

    def refresh_chat_view(self) -> None:
        model = safe_text(self.active_model_var.get())
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", tk.END)
        if not model or model not in self.config_data.histories:
            self.chat_text.insert(tk.END, "Select or add a model first.\n\n", "system_body")
        else:
            memory = self.config_data.histories[model]
            if not memory.messages:
                self.chat_text.insert(tk.END, "No messages yet.\n\n", "system_body")
            else:
                for item in memory.messages:
                    if item["role"] == "user":
                        self._append_chat_block("You", item["content"], "user_label", "user_body")
                    elif item["role"] == "assistant":
                        self._append_chat_block(
                            "Assistant", item["content"], "assistant_label", "assistant_body"
                        )
        self.chat_text.configure(state="disabled")
        self.chat_text.see(tk.END)
        self._refresh_meta()

    def _on_active_model_selected(self, event: tk.Event | None = None) -> None:
        if self._suspend_model_events:
            return
        self._preview_model_selection()

    def _on_summary_model_selected(self, event: tk.Event | None = None) -> None:
        if self._suspend_model_events:
            return
        self._preview_model_selection()

    def _on_model_list_select(self, event: tk.Event | None = None) -> None:
        if self._suspend_model_events:
            return
        self._preview_model_selection()

    def _preview_model_selection(self) -> None:
        selection = self.model_listbox.curselection()
        if selection:
            models = self._models()
            index = selection[0]
            if index < len(models):
                self.active_model_var.set(models[index])
        self._refresh_meta()

    def apply_model_fields(self) -> None:
        active_model = safe_text(self.active_model_var.get())
        summary_model = safe_text(self.summary_model_var.get()) or active_model
        if not active_model:
            self.status_var.set("Please enter a chat model.")
            return
        models = self._models()
        if active_model not in models:
            models.append(active_model)
        if summary_model and summary_model not in models:
            models.append(summary_model)
        self.config_data.models = unique_strings(models)
        self.config_data.active_model = active_model
        self.config_data.summary_model = summary_model
        self.config_data.histories.setdefault(active_model, ModelMemory())
        if summary_model:
            self.config_data.histories.setdefault(summary_model, ModelMemory())
        self._populate_model_controls()
        self._set_model_selection(active_model)
        self.save_settings(status=f"Active model set to {active_model}")
        self.refresh_chat_view()
        self._refresh_memory_panel()

    def set_active_from_selection(self) -> None:
        selection = self.model_listbox.curselection()
        if not selection:
            self.status_var.set("Select a model first.")
            return
        models = self._models()
        index = selection[0]
        if index < len(models):
            self.active_model_var.set(models[index])
            self.apply_model_fields()

    def set_summary_from_selection(self) -> None:
        selection = self.model_listbox.curselection()
        if not selection:
            self.status_var.set("Select a model first.")
            return
        models = self._models()
        index = selection[0]
        if index < len(models):
            self.summary_model_var.set(models[index])
            self.apply_model_fields()

    def add_model(self) -> None:
        model = safe_text(self.new_model_var.get())
        if not model:
            self.status_var.set("Enter a model name first.")
            return
        models = self._models()
        if model not in models:
            models.append(model)
        self.config_data.models = unique_strings(models)
        self.active_model_var.set(model)
        if not self.summary_model_var.get().strip():
            self.summary_model_var.set(model)
        self.config_data.histories.setdefault(model, ModelMemory())
        self.new_model_var.set("")
        self._populate_model_controls()
        self._set_model_selection(model)
        self.apply_model_fields()

    def delete_selected_model(self) -> None:
        selection = self.model_listbox.curselection()
        if not selection:
            self.status_var.set("Select a model to delete.")
            return
        models = self._models()
        index = selection[0]
        if index >= len(models):
            return
        model = models[index]
        if not messagebox.askyesno("Delete model", f"Delete '{model}'?"):
            return
        models.pop(index)
        self.config_data.models = unique_strings(models)
        self.config_data.histories.pop(model, None)
        if self.active_model_var.get().strip() == model:
            new_active = models[0] if models else ""
            self.active_model_var.set(new_active)
            self.config_data.active_model = new_active
        if self.summary_model_var.get().strip() == model:
            self.summary_model_var.set(self.active_model_var.get().strip())
            self.config_data.summary_model = self.summary_model_var.get().strip()
        self._populate_model_controls()
        if self.active_model_var.get().strip():
            self._set_model_selection(self.active_model_var.get().strip())
        self.save_settings(status=f"Deleted model: {model}")
        self.refresh_chat_view()
        self._refresh_memory_panel()

    def _template_names(self) -> list[str]:
        return list(self.config_data.templates.keys())

    def _on_template_selected(self, event: tk.Event | None = None) -> None:
        if self._suspend_template_events:
            return
        self.load_template_into_editor(save_active=True)

    def load_template_into_editor(self, save_active: bool = True) -> None:
        template = safe_text(self.active_template_var.get()) or "General"
        body = self.config_data.templates.get(
            template, DEFAULT_TEMPLATES.get(template, DEFAULT_SYSTEM_PROMPT)
        )
        self.template_name_var.set(template)
        self.template_text.configure(state="normal")
        self.template_text.delete("1.0", tk.END)
        self.template_text.insert(tk.END, body)
        self.config_data.active_template = template
        if save_active:
            self.save_settings(status=f"Loaded template: {template}")
        self._refresh_meta()

    def save_template(self) -> None:
        name = safe_text(self.template_name_var.get())
        body = safe_text(self.template_text.get("1.0", "end-1c"))
        if not name:
            self.status_var.set("Enter a template name first.")
            return
        if not body:
            self.status_var.set("Template body cannot be empty.")
            return
        self.config_data.templates[name] = body
        self.config_data.active_template = name
        self.active_template_var.set(name)
        self._populate_template_controls()
        self._set_template_selection(name)
        self.save_settings(status=f"Saved template: {name}")

    def new_template(self) -> None:
        self.active_template_var.set("General")
        self.template_name_var.set("")
        self.template_text.configure(state="normal")
        self.template_text.delete("1.0", tk.END)
        self.template_text.insert(tk.END, DEFAULT_SYSTEM_PROMPT)
        self.save_settings(status="New template prepared.")

    def delete_template(self) -> None:
        name = safe_text(self.template_name_var.get() or self.active_template_var.get())
        if not name:
            self.status_var.set("Select a template first.")
            return
        if name in DEFAULT_TEMPLATES:
            messagebox.showinfo("Not allowed", "Built-in templates cannot be deleted.")
            return
        if name not in self.config_data.templates:
            self.status_var.set("Template not found.")
            return
        if not messagebox.askyesno("Delete template", f"Delete '{name}'?"):
            return
        self.config_data.templates.pop(name, None)
        self._populate_template_controls()
        fallback = "General"
        self.active_template_var.set(fallback)
        self.template_name_var.set(fallback)
        self.load_template_into_editor(save_active=False)
        self.save_settings(status=f"Deleted template: {name}")

    def _set_template_selection(self, template: str) -> None:
        if template not in self.config_data.templates:
            return
        self._suspend_template_events = True
        try:
            self.template_combo.set(template)
        finally:
            self._suspend_template_events = False

    def save_params(self, status: str | None = "Params saved.") -> None:
        self.config_data.params = ChatParams(
            temperature=clamp_float(self.temperature_var.get(), 0.0, 2.0, 0.7),
            top_p=clamp_float(self.top_p_var.get(), 0.0, 1.0, 0.95),
            max_tokens=clamp_int(self.max_tokens_var.get(), 16, 32768, 2048),
            presence_penalty=clamp_float(self.presence_penalty_var.get(), -2.0, 2.0, 0.0),
            frequency_penalty=clamp_float(self.frequency_penalty_var.get(), -2.0, 2.0, 0.0),
        )
        self.temperature_var.set(str(self.config_data.params.temperature))
        self.top_p_var.set(str(self.config_data.params.top_p))
        self.max_tokens_var.set(str(self.config_data.params.max_tokens))
        self.presence_penalty_var.set(str(self.config_data.params.presence_penalty))
        self.frequency_penalty_var.set(str(self.config_data.params.frequency_penalty))
        self.save_settings(status=status)

    def reset_params(self) -> None:
        self.temperature_var.set("0.7")
        self.top_p_var.set("0.95")
        self.max_tokens_var.set("2048")
        self.presence_penalty_var.set("0.0")
        self.frequency_penalty_var.set("0.0")
        self.save_params("Reset to defaults.")

    def save_memory_settings(self) -> None:
        self.config_data.memory_settings = MemorySettings(
            auto_summary=bool(self.auto_summary_var.get()),
            summary_trigger_messages=clamp_int(self.summary_trigger_var.get(), 4, 200, 16),
            keep_recent_messages=clamp_int(self.keep_recent_var.get(), 2, 100, 8),
        )
        self.summary_trigger_var.set(str(self.config_data.memory_settings.summary_trigger_messages))
        self.keep_recent_var.set(str(self.config_data.memory_settings.keep_recent_messages))
        self.save_settings(status="Memory settings saved.")
        self._refresh_memory_panel()

    def clear_current_memory(self) -> None:
        model = safe_text(self.active_model_var.get())
        if not model:
            self.status_var.set("Select a model first.")
            return
        if not messagebox.askyesno("Clear memory", f"Clear memory for '{model}'?"):
            return
        memory = self._current_memory(model)
        memory.summary = ""
        memory.messages = []
        memory.summarized_upto = 0
        self.refresh_chat_view()
        self._refresh_memory_panel()
        self.save_settings(status=f"Cleared memory for {model}")

    def export_chat(self) -> None:
        model = safe_text(self.active_model_var.get())
        if not model:
            self.status_var.set("Select a model first.")
            return
        memory = self._current_memory(model)
        default_name = model.replace("/", "_").replace("\\", "_").replace(":", "_")
        path = filedialog.asksaveasfilename(
            title="Export chat",
            defaultextension=".md",
            initialfile=f"{default_name}_chat.md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        lines = [
            f"# Chat export: {model}",
            f"Template: {safe_text(self.active_template_var.get()) or 'General'}",
            f"Summary model: {safe_text(self.summary_model_var.get()) or model}",
            "",
        ]
        if memory.summary.strip():
            lines.extend(["## Summary", memory.summary.strip(), ""])
        lines.append("## Transcript")
        for item in memory.messages:
            role = "User" if item["role"] == "user" else "Assistant"
            lines.append(f"### {role}")
            lines.append(item["content"])
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        self.status_var.set(f"Exported chat to {path}")

    def _request_messages(self, model: str) -> list[dict[str, str]]:
        memory = self._current_memory(model)
        keep_recent = self.config_data.memory_settings.keep_recent_messages
        start_index = max(memory.summarized_upto, len(memory.messages) - keep_recent)
        messages: list[dict[str, str]] = [{"role": "system", "content": self._current_template_text()}]
        if memory.summary.strip():
            messages.append({"role": "system", "content": "Long-term memory:\n" + memory.summary.strip()})
        for item in memory.messages[start_index:]:
            messages.append({"role": item["role"], "content": item["content"]})
        return messages

    def _summary_segment(self, model: str) -> tuple[int, list[dict[str, str]]]:
        memory = self._current_memory(model)
        keep_recent = self.config_data.memory_settings.keep_recent_messages
        cutoff = max(memory.summarized_upto, len(memory.messages) - keep_recent)
        return cutoff, memory.messages[memory.summarized_upto:cutoff]

    def _needs_summary(self, model: str) -> bool:
        if not self.auto_summary_var.get():
            return False
        memory = self._current_memory(model)
        trigger = self.config_data.memory_settings.summary_trigger_messages
        cutoff = max(memory.summarized_upto, len(memory.messages) - self.config_data.memory_settings.keep_recent_messages)
        return len(memory.messages) >= trigger and cutoff > memory.summarized_upto

    def send_message(self) -> None:
        if self.chat_running or self.summary_running:
            return
        api_key = safe_text(self.api_key_var.get())
        if not api_key:
            self.status_var.set("API key is required.")
            return
        model = safe_text(self.active_model_var.get())
        if not model:
            self.status_var.set("Select or enter a chat model.")
            return
        user_text = safe_text(self.input_text.get("1.0", "end-1c"))
        if not user_text:
            return
        self.input_text.delete("1.0", tk.END)
        self._append_chat_block("You", user_text, "user_label", "user_body")
        self._start_assistant_block()

        request_id = uuid.uuid4().hex
        request_messages = self._request_messages(model) + [{"role": "user", "content": user_text}]
        self.chat_job_id = request_id
        self.pending_chat = {
            "id": request_id,
            "model": model,
            "user_text": user_text,
            "assistant_text": "",
            "start_time": time.perf_counter(),
        }
        self.chat_running = True
        self._update_control_state()
        self.status_var.set(f"Requesting {model} ...")

        thread = threading.Thread(
            target=self._chat_worker,
            args=(
                request_id,
                api_key,
                safe_text(self.base_url_var.get()) or DEFAULT_BASE_URL,
                model,
                request_messages,
                self.config_data.params,
            ),
            daemon=True,
        )
        thread.start()
        self._ensure_polling()

    def _chat_worker(
        self,
        request_id: str,
        api_key: str,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        params: ChatParams,
    ) -> None:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_tokens,
                presence_penalty=params.presence_penalty,
                frequency_penalty=params.frequency_penalty,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    self.event_queue.put({"kind": "chat_delta", "id": request_id, "text": content})
            self.event_queue.put({"kind": "chat_done", "id": request_id})
        except Exception as exc:
            self.event_queue.put({"kind": "chat_error", "id": request_id, "error": str(exc)})

    def summarize_now(self) -> None:
        model = safe_text(self.active_model_var.get())
        if not model:
            self.status_var.set("Select a model first.")
            return
        if self.summary_running:
            return
        cutoff, segment = self._summary_segment(model)
        if not segment:
            self.status_var.set("No new messages need summarizing.")
            return
        self._start_summary_job(model, segment, cutoff, reason="manual")

    def clear_current_chat(self) -> None:
        self.clear_current_memory()

    def _send_shortcut(self, event: tk.Event | None = None) -> str:
        self.send_message()
        return "break"

    def _start_summary_job(
        self, model: str, segment: list[dict[str, str]], cutoff: int, reason: str
    ) -> None:
        if self.summary_running:
            return
        request_id = uuid.uuid4().hex
        memory = self._current_memory(model)
        self.summary_job_id = request_id
        self.pending_summary = {
            "id": request_id,
            "model": model,
            "segment": segment,
            "cutoff": cutoff,
            "existing_summary": memory.summary.strip(),
            "reason": reason,
        }
        self.summary_running = True
        self._update_control_state()
        self.status_var.set("Summarizing memory ...")

        thread = threading.Thread(
            target=self._summary_worker,
            args=(
                request_id,
                safe_text(self.api_key_var.get()),
                safe_text(self.base_url_var.get()) or DEFAULT_BASE_URL,
                safe_text(self.summary_model_var.get()) or model,
                segment,
                memory.summary.strip(),
            ),
            daemon=True,
        )
        thread.start()
        self._ensure_polling()

    def _summary_worker(
        self,
        request_id: str,
        api_key: str,
        base_url: str,
        summary_model: str,
        segment: list[dict[str, str]],
        existing_summary: str,
    ) -> None:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            segment_text = "\n".join(
                f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}"
                for item in segment
            )
            prompt = (
                "You are a long-term memory compressor. Merge the existing summary and the new "
                "conversation into a compact, durable memory. Keep user preferences, project "
                "state, decisions, constraints, terminology, pending tasks, and durable facts. "
                "Drop greetings, repetition, and transient details. Write in the same language "
                "as the conversation. Keep it short, but do not lose important information."
            )
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Existing summary:\n{existing_summary or '(none)'}\n\n"
                        f"New conversation segment:\n{segment_text}\n\n"
                        "Return the updated summary only."
                    ),
                },
            ]
            response = client.chat.completions.create(
                model=summary_model,
                messages=messages,
                temperature=0.2,
                top_p=1.0,
                max_tokens=512,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
            text = ""
            if response.choices:
                text = safe_text(response.choices[0].message.content)
            if not text:
                raise RuntimeError("The summary model returned no content.")
            self.event_queue.put(
                {
                    "kind": "summary_done",
                    "id": request_id,
                    "model": self.pending_summary["model"] if self.pending_summary else "",
                    "summary": text,
                    "cutoff": self.pending_summary["cutoff"] if self.pending_summary else 0,
                    "segment_count": len(segment),
                }
            )
        except Exception as exc:
            self.event_queue.put({"kind": "summary_error", "id": request_id, "error": str(exc)})

    def _ensure_polling(self) -> None:
        if self.poll_scheduled:
            return
        self.poll_scheduled = True
        self.after(30, self._poll_queue)

    def _poll_queue(self) -> None:
        self.poll_scheduled = False
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            kind = event.get("kind")
            event_id = event.get("id")
            if kind == "chat_delta" and event_id == self.chat_job_id and self.pending_chat:
                text = safe_text(event.get("text", ""))
                if text:
                    self.pending_chat["assistant_text"] = (
                        safe_text(self.pending_chat.get("assistant_text", "")) + text
                    )
                    self.chat_text.configure(state="normal")
                    self.chat_text.insert(tk.END, text, "assistant_body")
                    self.chat_text.configure(state="disabled")
                    self.chat_text.see(tk.END)
            elif kind == "chat_done" and event_id == self.chat_job_id and self.pending_chat:
                assistant_text = safe_text(self.pending_chat.get("assistant_text", ""))
                self.chat_text.configure(state="normal")
                self.chat_text.insert(tk.END, "\n\n", "assistant_body")
                self.chat_text.configure(state="disabled")
                self.chat_text.see(tk.END)
                self._store_chat_turn(self.pending_chat["model"], self.pending_chat["user_text"], assistant_text)
                self.last_latency = time.perf_counter() - float(self.pending_chat["start_time"])
                model = self.pending_chat["model"]
                self.chat_running = False
                self.chat_job_id = None
                self.pending_chat = None

                if self._needs_summary(model):
                    cutoff, segment = self._summary_segment(model)
                    self._start_summary_job(model, segment, cutoff, reason="auto")
                self.save_settings(status=None)
                self.status_var.set(
                    f"Completed in {self.last_latency:.2f}s. {self._status_suffix()}"
                )
            elif kind == "chat_error" and event_id == self.chat_job_id:
                error = safe_text(event.get("error", ""))
                self._append_chat_block("Error", error, "error_label", "error_body")
                self.chat_running = False
                self.chat_job_id = None
                self.pending_chat = None
                self.status_var.set(f"Request failed: {error}")
            elif kind == "summary_done" and event_id == self.summary_job_id and self.pending_summary:
                model = safe_text(event.get("model", ""))
                summary = safe_text(event.get("summary", ""))
                cutoff = int(event.get("cutoff", 0) or 0)
                if model and summary:
                    memory = self._current_memory(model)
                    memory.summary = self._merge_summary(memory.summary, summary)
                    memory.summarized_upto = max(memory.summarized_upto, cutoff)
                    self._refresh_memory_panel()
                    self.refresh_chat_view()
                    self.save_settings(status=None)
                    self.status_var.set(
                        f"Memory compressed: {int(event.get('segment_count', 0))} messages. {self._status_suffix()}"
                    )
                self.summary_running = False
                self.summary_job_id = None
                self.pending_summary = None
            elif kind == "summary_error" and event_id == self.summary_job_id:
                error = safe_text(event.get("error", ""))
                self.summary_running = False
                self.summary_job_id = None
                self.pending_summary = None
                self.status_var.set(f"Summary failed: {error}")

        self._update_control_state()
        self._refresh_meta()
        if self.chat_running or self.summary_running or not self.event_queue.empty():
            self._ensure_polling()

    def _store_chat_turn(self, model: str, user_text: str, assistant_text: str) -> None:
        memory = self._current_memory(model)
        memory.messages.append({"role": "user", "content": user_text})
        memory.messages.append({"role": "assistant", "content": assistant_text})

    def _merge_summary(self, old_summary: str, new_summary: str) -> str:
        old_summary = old_summary.strip()
        new_summary = new_summary.strip()
        if not old_summary:
            return new_summary
        if not new_summary:
            return old_summary
        return old_summary + "\n" + new_summary

    def _status_suffix(self) -> str:
        model = safe_text(self.active_model_var.get()) or "None"
        summary_model = safe_text(self.summary_model_var.get()) or model
        template = safe_text(self.active_template_var.get()) or "General"
        return f"| Chat {model} | Summary {summary_model} | Template {template}"

    def save_settings(self, status: str | None = "Settings saved.") -> None:
        self.config_data.api_key = safe_text(self.api_key_var.get())
        self.config_data.base_url = safe_text(self.base_url_var.get()) or DEFAULT_BASE_URL
        self.config_data.active_model = safe_text(self.active_model_var.get())
        self.config_data.summary_model = safe_text(self.summary_model_var.get())
        self.config_data.active_template = safe_text(self.active_template_var.get()) or "General"
        self.config_data.geometry = self.geometry()
        self.config_data.params = ChatParams(
            temperature=clamp_float(self.temperature_var.get(), 0.0, 2.0, 0.7),
            top_p=clamp_float(self.top_p_var.get(), 0.0, 1.0, 0.95),
            max_tokens=clamp_int(self.max_tokens_var.get(), 16, 32768, 2048),
            presence_penalty=clamp_float(self.presence_penalty_var.get(), -2.0, 2.0, 0.0),
            frequency_penalty=clamp_float(self.frequency_penalty_var.get(), -2.0, 2.0, 0.0),
        )
        self.config_data.memory_settings = MemorySettings(
            auto_summary=bool(self.auto_summary_var.get()),
            summary_trigger_messages=clamp_int(self.summary_trigger_var.get(), 4, 200, 16),
            keep_recent_messages=clamp_int(self.keep_recent_var.get(), 2, 100, 8),
        )
        self.config_data.models = unique_strings(
            self._models() + [self.config_data.active_model, self.config_data.summary_model]
        )
        for model in self.config_data.models:
            self.config_data.histories.setdefault(model, ModelMemory())
        self._populate_model_controls()
        self._populate_template_controls()
        self._refresh_meta()
        self._update_control_state()
        try:
            save_config(self.config_data)
        except Exception:
            pass
        if status is not None:
            self.status_var.set(status)

    def refresh_all_views(self) -> None:
        self.refresh_chat_view()
        self._refresh_memory_panel()
        self._refresh_meta()

    def on_close(self) -> None:
        self.save_settings(status=None)
        self.destroy()


def main() -> int:
    app = SiliconFlowChatApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
