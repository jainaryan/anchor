"""
Anchor pitch deck generator.
Creates a modern, minimal 16:9 deck using python-pptx.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

# ----- Design tokens -----
BG = RGBColor(0x0E, 0x14, 0x14)          # deep near-black
BG_PANEL = RGBColor(0x14, 0x1C, 0x1C)    # card panel
ACCENT = RGBColor(0x7E, 0xAA, 0xA0)      # sage/teal (logo)
ACCENT_DIM = RGBColor(0x4E, 0x6F, 0x68)
TEXT = RGBColor(0xEA, 0xEF, 0xED)
MUTED = RGBColor(0x95, 0xA3, 0xA0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DIVIDER = RGBColor(0x26, 0x33, 0x31)

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, "anchor_logo.png")
OUT = os.path.join(HERE, "Anchor_Pitch_Deck.pptx")

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]


# ----- helpers -----
def add_bg(slide, color=BG):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_rect(slide, x, y, w, h, fill=BG_PANEL, line=None, corner=False):
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if corner else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    if corner:
        try:
            shp.adjustments[0] = 0.08
        except Exception:
            pass
    return shp


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=TEXT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Inter",
             line_spacing=1.2):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return tb


def add_runs(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, font="Inter", line_spacing=1.25,
             default_size=16, default_color=TEXT):
    """runs: list of paragraphs; each paragraph is list of (text, {size,bold,color,italic})"""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        if isinstance(para, str):
            para = [(para, {})]
        for (txt, attrs) in para:
            r = p.add_run()
            r.text = txt
            r.font.name = attrs.get("font", font)
            r.font.size = Pt(attrs.get("size", default_size))
            r.font.bold = attrs.get("bold", False)
            r.font.italic = attrs.get("italic", False)
            r.font.color.rgb = attrs.get("color", default_color)
    return tb


def add_footer(slide, n, total, title):
    # thin divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6),
                                  SLIDE_H - Inches(0.55),
                                  SLIDE_W - Inches(1.2), Emu(8000))
    line.fill.solid()
    line.fill.fore_color.rgb = DIVIDER
    line.line.fill.background()
    line.shadow.inherit = False

    add_text(slide, Inches(0.6), SLIDE_H - Inches(0.42),
             Inches(4), Inches(0.3),
             "ANCHOR", size=9, bold=True, color=ACCENT)
    add_text(slide, Inches(4.6), SLIDE_H - Inches(0.42),
             Inches(4.1), Inches(0.3), title, size=9, color=MUTED,
             align=PP_ALIGN.CENTER)
    add_text(slide, SLIDE_W - Inches(2), SLIDE_H - Inches(0.42),
             Inches(1.4), Inches(0.3), f"{n:02d} / {total:02d}",
             size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def add_logo(slide, x, y, size=Inches(0.55)):
    slide.shapes.add_picture(LOGO, x, y, width=size, height=size)


def add_kicker(slide, x, y, w, text):
    # accent dot + kicker label
    dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y + Inches(0.07),
                                 Inches(0.12), Inches(0.12))
    dot.fill.solid()
    dot.fill.fore_color.rgb = ACCENT
    dot.line.fill.background()
    dot.shadow.inherit = False
    add_text(slide, x + Inches(0.25), y, w, Inches(0.3),
             text.upper(), size=10, bold=True, color=ACCENT)


TOTAL = 11

# =====================================================================
# SLIDE 1 — Cover / Punchy intro
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
# Left: big copy
add_logo(s, Inches(0.6), Inches(0.55), Inches(0.7))
add_text(s, Inches(1.45), Inches(0.62), Inches(4), Inches(0.5),
         "ANCHOR", size=16, bold=True, color=TEXT)

add_runs(
    s, Inches(0.6), Inches(2.1), Inches(8.2), Inches(3.6),
    runs=[
        [("Mental health apps only work ", {"size": 40, "bold": True, "color": TEXT}),
         ("when you open them.", {"size": 40, "bold": True, "color": MUTED})],
        [("But struggling people ", {"size": 40, "bold": True, "color": TEXT}),
         ("don't open apps.", {"size": 40, "bold": True, "color": MUTED})],
        [("Anchor meets you ", {"size": 40, "bold": True, "color": TEXT}),
         ("before you break.", {"size": 40, "bold": True, "color": ACCENT})],
    ],
    line_spacing=1.15,
)

add_text(s, Inches(0.6), Inches(5.9), Inches(8), Inches(0.4),
         "Proactive. Private. Personalized.",
         size=16, color=MUTED)
add_text(s, Inches(0.6), Inches(6.25), Inches(8), Inches(0.4),
         "Pre-seed pitch  ·  2026", size=11, color=ACCENT_DIM)

# Right: large logo watermark
s.shapes.add_picture(LOGO, Inches(8.6), Inches(1.6),
                     width=Inches(4.2), height=Inches(4.2))

add_footer(s, 1, TOTAL, "Cover")

# =====================================================================
# SLIDE 2 — Problem
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
add_kicker(s, Inches(0.6), Inches(0.55), Inches(6), "The Problem")
add_text(s, Inches(0.6), Inches(0.9), Inches(12), Inches(0.8),
         "Mental health support is broken at every layer.",
         size=30, bold=True, color=TEXT)

# 4 problem cards in a 2x2 grid
cards = [
    ("Massive Unmet Need",
     [
         "~1 billion people live with a mental disorder",
         "1 in 8 affected globally (WHO, 2019)",
         ">70% receive no treatment (Thornicroft, AJPH 2013)",
         "1 in 2 will experience a disorder in their lifetime (Lancet Psychiatry, 2024)",
     ]),
    ("Access Is Fundamentally Broken",
     [
         "3+ month waitlists for appointments",
         "56% of psychologists not accepting new patients (APA, 2023)",
         "$150–$300 per session out-of-pocket (SimplePractice, 2024)",
         "Slow, expensive, and out of reach when it matters",
     ]),
    ("Existing Solutions Fail In Practice",
     [
         "Apps are reactive — you must initiate help",
         "No proactive detection of distress",
         "Built for engagement, not outcomes",
         "Headspace, Calm, BetterHelp all wait for you",
     ]),
    ("Trust Is Collapsing",
     [
         "72% of mental health apps share data with 3rd parties (2023)",
         "77% of adults worry about sharing health data with AI (KFF, 2026)",
         "Users don't trust the systems meant to help them",
         "Privacy concerns are now a conversion blocker",
     ]),
]

grid_left = Inches(0.6)
grid_top = Inches(1.85)
card_w = Inches(6.05)
card_h = Inches(2.55)
gap_x = Inches(0.22)
gap_y = Inches(0.22)

for idx, (title, bullets) in enumerate(cards):
    row, col = divmod(idx, 2)
    x = grid_left + (card_w + gap_x) * col
    y = grid_top + (card_h + gap_y) * row
    add_rect(s, x, y, card_w, card_h, fill=BG_PANEL, corner=True)
    # accent bar
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x + Inches(0.25),
                             y + Inches(0.25), Inches(0.1), Inches(0.35))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background(); bar.shadow.inherit = False
    add_text(s, x + Inches(0.45), y + Inches(0.2), card_w - Inches(0.6),
             Inches(0.45), title, size=16, bold=True, color=TEXT)
    # bullets
    paras = [[("•  ", {"color": ACCENT, "bold": True, "size": 12}),
              (b, {"size": 12, "color": MUTED})] for b in bullets]
    add_runs(s, x + Inches(0.45), y + Inches(0.85),
             card_w - Inches(0.7), card_h - Inches(1.0),
             runs=paras, line_spacing=1.25, default_size=12,
             default_color=MUTED)

add_footer(s, 2, TOTAL, "The Problem")

# =====================================================================
# SLIDE 3 — What we're doing differently
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
add_kicker(s, Inches(0.6), Inches(0.55), Inches(6), "What We Do Differently")
add_text(s, Inches(0.6), Inches(0.9), Inches(12), Inches(0.8),
         "The app comes to you, informed — not the other way around.",
         size=26, bold=True, color=TEXT)

diff = [
    ("01",
     "We detect distress before you ask for help",
     "Anchor passively monitors HRV, sleep, and activity. An on-device classifier catches early signs of emotional distress and triggers a proactive check-in. No other consumer app does this — every competitor waits for you to open it."),
    ("02",
     "Your data never leaves your phone",
     "The AI runs entirely on-device. No cloud inference. No data upload. No third-party exposure. Privacy isn't a feature we bolted on — it's the architecture."),
    ("03",
     "The AI is context-aware, not generic",
     "Models are fine-tuned on mental health conversations and aligned via DPO for safe, supportive responses. Biometric state is injected into the system prompt at inference — so the AI already knows how you're doing before you say a word."),
]

col_w = Inches(4.0)
col_h = Inches(4.5)
col_left = Inches(0.6)
col_top = Inches(1.95)
for i, (num, title, body) in enumerate(diff):
    x = col_left + (col_w + Inches(0.25)) * i
    add_rect(s, x, col_top, col_w, col_h, fill=BG_PANEL, corner=True)
    add_text(s, x + Inches(0.35), col_top + Inches(0.25), col_w - Inches(0.5),
             Inches(0.7), num, size=36, bold=True, color=ACCENT)
    add_text(s, x + Inches(0.35), col_top + Inches(1.0),
             col_w - Inches(0.5), Inches(1.2), title,
             size=17, bold=True, color=TEXT, line_spacing=1.15)
    add_text(s, x + Inches(0.35), col_top + Inches(2.25),
             col_w - Inches(0.5), col_h - Inches(2.4), body,
             size=12, color=MUTED, line_spacing=1.35)

add_text(s, Inches(0.6), Inches(6.55), Inches(12.2), Inches(0.5),
         "“Proactive. Private. Personalized — not because it sounds good, "
         "but because the architecture demands it.”",
         size=12, color=ACCENT, align=PP_ALIGN.CENTER)

add_footer(s, 3, TOTAL, "Differentiation")

# =====================================================================
# SLIDE 4 — Market Potential
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
add_kicker(s, Inches(0.6), Inches(0.55), Inches(6), "Market")
add_text(s, Inches(0.6), Inches(0.9), Inches(12), Inches(0.8),
         "A massive market, accelerating fast.",
         size=28, bold=True, color=TEXT)

# TAM / SAM / SOM cards
tiers = [
    ("TAM", "Digital Mental Health",
     "$27.84B", "2024",
     "→ $153.03B by 2034",
     "18.58% CAGR · 70% global treatment gap"),
    ("SAM", "Mental Health Apps + AI",
     "$7.48B", "2024",
     "→ $17.52B by 2030",
     "AI-in-MH: $1.45B → $11.84B (24.15% CAGR)"),
    ("SOM", "Young Adults 18–35",
     "600K", "users @ 0.2%",
     "= $36M ARR at $5/mo",
     "US · UK · Singapore · India"),
]
tier_w = Inches(4.0)
tier_h = Inches(2.6)
tier_top = Inches(1.9)
for i, (label, cat, big, small, delta, sub) in enumerate(tiers):
    x = Inches(0.6) + (tier_w + Inches(0.25)) * i
    add_rect(s, x, tier_top, tier_w, tier_h, fill=BG_PANEL, corner=True)
    add_text(s, x + Inches(0.35), tier_top + Inches(0.2),
             tier_w - Inches(0.5), Inches(0.35), label, size=11,
             bold=True, color=ACCENT)
    add_text(s, x + Inches(0.35), tier_top + Inches(0.55),
             tier_w - Inches(0.5), Inches(0.35), cat, size=13,
             color=MUTED)
    add_text(s, x + Inches(0.35), tier_top + Inches(1.0),
             tier_w - Inches(0.5), Inches(0.7), big, size=32,
             bold=True, color=TEXT)
    add_text(s, x + Inches(0.35), tier_top + Inches(1.7),
             tier_w - Inches(0.5), Inches(0.3), f"{small}  {delta}",
             size=11, color=ACCENT)
    add_text(s, x + Inches(0.35), tier_top + Inches(2.05),
             tier_w - Inches(0.5), Inches(0.45), sub, size=11,
             color=MUTED)

# Why now row
wn_top = Inches(4.75)
add_text(s, Inches(0.6), wn_top, Inches(6), Inches(0.4),
         "Why Now", size=14, bold=True, color=ACCENT)
why_now = [
    ("Hardware threshold", "On-device AI inference is finally viable. No more privacy ↔ intelligence trade-off."),
    ("Wearable ubiquity", "500M+ smartwatch users globally. Passive biometric tracking is mainstream."),
    ("New baseline", "The 25% global spike in anxiety never receded — the market is pulling for proactive care."),
]
wn_w = Inches(4.0)
wn_h = Inches(1.75)
wn_box_top = Inches(5.2)
for i, (t, b) in enumerate(why_now):
    x = Inches(0.6) + (wn_w + Inches(0.25)) * i
    add_rect(s, x, wn_box_top, wn_w, wn_h, fill=BG_PANEL, corner=True)
    add_text(s, x + Inches(0.3), wn_box_top + Inches(0.2),
             wn_w - Inches(0.5), Inches(0.4), t, size=13, bold=True,
             color=TEXT)
    add_text(s, x + Inches(0.3), wn_box_top + Inches(0.65),
             wn_w - Inches(0.5), wn_h - Inches(0.75), b, size=11,
             color=MUTED, line_spacing=1.3)

add_text(s, Inches(0.6), Inches(7.05), Inches(12.2), Inches(0.3),
         "Sources: Towards Healthcare / Precedence Research 2025 · Grand View Research 2025 · GlobeNewswire 2025",
         size=8, color=ACCENT_DIM)

add_footer(s, 4, TOTAL, "Market")

# =====================================================================
# SLIDE 5 — Competitors
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
add_kicker(s, Inches(0.6), Inches(0.55), Inches(6), "Landscape")
add_text(s, Inches(0.6), Inches(0.9), Inches(12), Inches(0.8),
         "The landscape is crowded — but empty where it counts.",
         size=26, bold=True, color=TEXT)

# Build a comparison table header + rows
tbl_left = Inches(0.6)
tbl_top = Inches(1.95)
tbl_w = Inches(12.1)
header_h = Inches(0.5)
row_h = Inches(0.74)

# Header bar
add_rect(s, tbl_left, tbl_top, tbl_w, header_h, fill=BG_PANEL, corner=False)
cols = [("Category", Inches(3.1)),
        ("Examples", Inches(2.3)),
        ("The Flaw", Inches(4.5)),
        ("Proactive?", Inches(1.1)),
        ("Private?", Inches(1.1))]
x_cursor = tbl_left + Inches(0.25)
for title, w in cols:
    add_text(s, x_cursor, tbl_top + Inches(0.1), w, Inches(0.35),
             title, size=10, bold=True, color=ACCENT)
    x_cursor += w

rows = [
    ("Traditional Teletherapy", "BetterHelp, Talkspace",
     "Reactive, high friction, prohibitively expensive. Days of waiting for real-time need.",
     "No", "Partial"),
    ("Meditation & Wellness", "Calm, Headspace",
     "Reactive and generic. You must realize you're stressed and choose a track.",
     "No", "Partial"),
    ("Legacy AI Chatbots", "Wysa, Woebot",
     "Cloud-hosted (privacy risk) and scripted. Feels like a phone-menu, not support.",
     "No", "No"),
    ("AI Companions", "Replika",
     "Built for addiction, not wellbeing. Cloud-dependent. No therapeutic guardrails.",
     "No", "No"),
]
row_y = tbl_top + header_h
for i, row in enumerate(rows):
    fill = BG if i % 2 == 0 else BG_PANEL
    add_rect(s, tbl_left, row_y, tbl_w, row_h, fill=fill)
    x_cursor = tbl_left + Inches(0.25)
    for (val, (_, w)) in zip(row, cols):
        color = TEXT if w > Inches(2) else MUTED
        size = 11 if w > Inches(2) else 10
        bold = (w == Inches(3.1))
        if val in ("No",):
            color = RGBColor(0xE8, 0x7D, 0x7D)
            bold = True
        if val == "Partial":
            color = RGBColor(0xE0, 0xB5, 0x6B)
            bold = True
        add_text(s, x_cursor, row_y + Inches(0.15), w - Inches(0.15),
                 row_h - Inches(0.1), val, size=size, bold=bold,
                 color=color, line_spacing=1.2)
        x_cursor += w
    row_y += row_h

# Anchor highlight row
anchor_y = row_y + Inches(0.15)
add_rect(s, tbl_left, anchor_y, tbl_w, row_h + Inches(0.15),
         fill=BG_PANEL, line=ACCENT, corner=True)
x_cursor = tbl_left + Inches(0.25)
anchor_row = ("Anchor", "— The only one",
              "Reaches out when biometrics spike. 100% on-device. DPO-aligned for grounding & crisis de-escalation.",
              "Yes", "Yes")
for val, (_, w) in zip(anchor_row, cols):
    color = ACCENT
    size = 12 if w > Inches(2) else 11
    bold = True
    add_text(s, x_cursor, anchor_y + Inches(0.22), w - Inches(0.15),
             row_h, val, size=size, bold=bold, color=color,
             line_spacing=1.2)
    x_cursor += w

add_footer(s, 5, TOTAL, "Competitive Landscape")

# =====================================================================
# SLIDE 6 — Technical Moat
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
add_kicker(s, Inches(0.6), Inches(0.55), Inches(6), "Why We're Better")
add_text(s, Inches(0.6), Inches(0.9), Inches(12), Inches(0.8),
         "We aren't a wrapper. Hard engineering is our defensibility.",
         size=26, bold=True, color=TEXT)

moats = [
    ("Predictive Moat",
     "Real-time biometric ML",
     "A continuous on-device classifier on HRV, sleep, and activity predicts emotional spikes before they peak — without draining the battery."),
    ("Infrastructure Moat",
     "Local LLM orchestration",
     "Optimized Llama models run 100% on-device alongside the predictive ML classifier and real-time ingestion. Zero cloud, zero latency, absolute privacy."),
    ("Context Moat",
     "On-device memory engine",
     "A persistent local memory retrieval engine references past struggles and growth — longitudinal context, never exposed to a server."),
    ("Efficacy Moat",
     "Proprietary fine-tuning + DPO",
     "Models aligned via custom data pipelines and Direct Preference Optimization for crisis de-escalation and grounding — not generic chat."),
]
m_left = Inches(0.6)
m_top = Inches(1.9)
m_w = Inches(6.05)
m_h = Inches(2.3)
gap = Inches(0.22)
for i, (label, title, body) in enumerate(moats):
    row, col = divmod(i, 2)
    x = m_left + (m_w + gap) * col
    y = m_top + (m_h + gap) * row
    add_rect(s, x, y, m_w, m_h, fill=BG_PANEL, corner=True)
    add_text(s, x + Inches(0.35), y + Inches(0.2), m_w - Inches(0.5),
             Inches(0.35), label.upper(), size=10, bold=True, color=ACCENT)
    add_text(s, x + Inches(0.35), y + Inches(0.55), m_w - Inches(0.5),
             Inches(0.5), title, size=17, bold=True, color=TEXT)
    add_text(s, x + Inches(0.35), y + Inches(1.05), m_w - Inches(0.55),
             m_h - Inches(1.15), body, size=12, color=MUTED,
             line_spacing=1.35)

add_footer(s, 6, TOTAL, "Technical Moat")

# =====================================================================
# SLIDE 7 — Team
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
add_kicker(s, Inches(0.6), Inches(0.55), Inches(6), "The Team")
add_text(s, Inches(0.6), Inches(0.9), Inches(12), Inches(0.8),
         "Two founders. One decade of shared execution.",
         size=26, bold=True, color=TEXT)

# Two founder cards
f_top = Inches(1.95)
f_h = Inches(4.1)
f_w = Inches(6.05)

def founder_card(slide, x, y, w, h, name, role, badges, bullets):
    add_rect(slide, x, y, w, h, fill=BG_PANEL, corner=True)
    # avatar circle
    av = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.35),
                                y + Inches(0.35), Inches(0.8), Inches(0.8))
    av.fill.solid(); av.fill.fore_color.rgb = ACCENT
    av.line.fill.background(); av.shadow.inherit = False
    initials = "".join([p[0] for p in name.split()[:2]])
    add_text(slide, x + Inches(0.35), y + Inches(0.45),
             Inches(0.8), Inches(0.6), initials, size=22, bold=True,
             color=BG, align=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(1.35), y + Inches(0.35),
             w - Inches(1.5), Inches(0.45), name, size=20, bold=True,
             color=TEXT)
    add_text(slide, x + Inches(1.35), y + Inches(0.75),
             w - Inches(1.5), Inches(0.4), role, size=12, color=ACCENT)
    # badges row
    bx = x + Inches(0.35)
    by = y + Inches(1.4)
    for b in badges:
        approx_w = Inches(0.22 + 0.09 * len(b))
        add_rect(slide, bx, by, approx_w, Inches(0.32),
                 fill=BG, line=ACCENT_DIM, corner=True)
        add_text(slide, bx, by + Inches(0.05), approx_w, Inches(0.28),
                 b, size=9, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
        bx += approx_w + Inches(0.08)
    # bullets
    paras = [[("— ", {"color": ACCENT, "bold": True, "size": 12}),
              (b, {"size": 12, "color": MUTED})] for b in bullets]
    add_runs(slide, x + Inches(0.35), y + Inches(1.95),
             w - Inches(0.7), h - Inches(2.1),
             runs=paras, line_spacing=1.3, default_size=12,
             default_color=MUTED)

founder_card(
    s, Inches(0.6), f_top, f_w, f_h,
    "Aryan Jain", "Founder & CEO",
    ["NUS CVML", "IIT-G", "3 papers", "Chess Top-20 IN"],
    [
        "Deep expertise in VLMs, Reinforcement Learning, and Graph ML",
        "Researcher at NUS CVML lab; formerly at IIT Guwahati",
        "Collaborating with Stanford postdocs on DL optimizations",
        "3 published papers (IEEE / Springer); tier-1 journal reviewer",
        "Top-performer across Indian Olympiads",
    ],
)
founder_card(
    s, Inches(0.6) + f_w + Inches(0.25), f_top, f_w, f_h,
    "Tanmay Kukreja", "Co-Founder & CTO",
    ["Mercedes-Benz R&D", "Deloitte", "Hitachi", "Time-Series"],
    [
        "Data Scientist at Mercedes-Benz R&D — deep time-series expertise",
        "Exact architecture needed for real-time HRV/sleep streams",
        "Applied LLM engineering background from research tenure at Hitachi",
        "State-level badminton + school athlete — competitive execution DNA",
    ],
)

# unfair advantage strip
ua_y = Inches(6.2)
add_rect(s, Inches(0.6), ua_y, Inches(12.1), Inches(0.85),
         fill=BG_PANEL, line=ACCENT, corner=True)
add_text(s, Inches(0.85), ua_y + Inches(0.12), Inches(11.6),
         Inches(0.3), "OUR UNFAIR ADVANTAGE", size=10, bold=True,
         color=ACCENT)
add_text(s, Inches(0.85), ua_y + Inches(0.4), Inches(11.6),
         Inches(0.5),
         "Zero co-founder risk — years of building together, a co-authored paper, "
         "and both All-India Toppers with 100/100 in CBSE Mathematics.",
         size=12, color=TEXT)

add_footer(s, 7, TOTAL, "Team")

# =====================================================================
# SLIDE 8 — Roadmap
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
add_kicker(s, Inches(0.6), Inches(0.55), Inches(6), "Roadmap · Next 12 Months")
add_text(s, Inches(0.6), Inches(0.9), Inches(12), Inches(0.8),
         "Build the moat, then scale the wedge.",
         size=26, bold=True, color=TEXT)

# Timeline bar
tl_y = Inches(2.2)
add_rect(s, Inches(0.6), tl_y, Inches(12.1), Inches(0.08),
         fill=DIVIDER)
# two phase markers
for i, x in enumerate([Inches(0.6), Inches(6.8)]):
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, x - Inches(0.12),
                             tl_y - Inches(0.1), Inches(0.3), Inches(0.3))
    dot.fill.solid(); dot.fill.fore_color.rgb = ACCENT
    dot.line.fill.background(); dot.shadow.inherit = False

# Phase 1 card
p1x = Inches(0.6); p1y = Inches(2.75)
add_rect(s, p1x, p1y, Inches(5.9), Inches(4.2), fill=BG_PANEL, corner=True)
add_text(s, p1x + Inches(0.35), p1y + Inches(0.25), Inches(5.3),
         Inches(0.4), "PHASE 1  ·  MONTHS 1–6", size=10, bold=True,
         color=ACCENT)
add_text(s, p1x + Inches(0.35), p1y + Inches(0.62), Inches(5.3),
         Inches(0.5), "Core Engineering & Closed Beta",
         size=19, bold=True, color=TEXT)
p1_bullets = [
    ("Model optimization",
     "Test new localized inference methods and stress-test across older smartphone hardware."),
    ("Voice stack exploration",
     "Prototype integrating STT + TTS models for voice-based check-ins."),
    ("Wearable architecture",
     "Robust pipelines for real-time biometric ingestion across diverse devices."),
    ("App refinement",
     "Battery, memory retrieval, and UI/UX polish."),
    ("Closed beta",
     "Ship to early adopters ASAP to validate predictive ML triggers in the wild."),
]
y = p1y + Inches(1.3)
for title, body in p1_bullets:
    add_text(s, p1x + Inches(0.35), y, Inches(5.3), Inches(0.28),
             "•  " + title, size=12, bold=True, color=ACCENT)
    add_text(s, p1x + Inches(0.55), y + Inches(0.3), Inches(5.1),
             Inches(0.32), body, size=10.5, color=MUTED, line_spacing=1.25)
    y += Inches(0.56)

# Phase 2 card
p2x = Inches(6.8); p2y = Inches(2.75)
add_rect(s, p2x, p2y, Inches(5.9), Inches(4.2), fill=BG_PANEL, corner=True)
add_text(s, p2x + Inches(0.35), p2y + Inches(0.25), Inches(5.3),
         Inches(0.4), "PHASE 2  ·  MONTHS 6–12", size=10, bold=True,
         color=ACCENT)
add_text(s, p2x + Inches(0.35), p2y + Inches(0.62), Inches(5.3),
         Inches(0.5), "Go-to-Market & Visibility",
         size=19, bold=True, color=TEXT)
p2_bullets = [
    ("Aggressive marketing push",
     "Shift from R&D to user acquisition, targeting high-stress young adults."),
    ("Events & community",
     "Hosting and sponsoring targeted events in the Bengaluru startup ecosystem."),
    ("Strategic collaborations",
     "Partner with universities, tech hubs, and wearable communities for organic growth."),
    ("Scale to 10K users",
     "Validate the freemium flip and prepare the Series-seed narrative."),
]
y = p2y + Inches(1.3)
for title, body in p2_bullets:
    add_text(s, p2x + Inches(0.35), y, Inches(5.3), Inches(0.28),
             "•  " + title, size=12, bold=True, color=ACCENT)
    add_text(s, p2x + Inches(0.55), y + Inches(0.3), Inches(5.1),
             Inches(0.32), body, size=10.5, color=MUTED, line_spacing=1.25)
    y += Inches(0.56)

add_footer(s, 8, TOTAL, "Roadmap")

# =====================================================================
# SLIDE 9 — The Ask / Use of Funds
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
add_kicker(s, Inches(0.6), Inches(0.55), Inches(6), "The Ask")
add_text(s, Inches(0.6), Inches(0.9), Inches(12), Inches(0.8),
         "Raising $250K pre-seed  ·  ~₹2.3 Cr  ·  18–24 months runway",
         size=24, bold=True, color=TEXT)

# Left column — bar allocation
alloc = [
    ("Top Talent & Core Team", "40%", "₹92L",
     "Mobile (CoreML/JNI) + UX hire, founder stipends over 24 mo."),
    ("R&D & Hardware Infrastructure", "17%", "₹39L",
     "Compute (₹34L) for DPO, fine-tuning, synthetic data. Device farm (₹5L)."),
    ("Legal, Compliance & Ops", "15%", "₹35L",
     "Enterprise AI stack (₹20L), privacy audits & legal (₹15L)."),
    ("Lean GTM & Acquisition", "10%", "₹23L",
     "Events & community (₹5L), targeted growth marketing (₹18L)."),
    ("Headway & Contingency", "18%", "₹41L",
     "War chest for R&D bottlenecks, runway extension, optional office in Yr 2."),
]
al_top = Inches(2.0)
al_w = Inches(8.0)
bar_h = Inches(0.28)
row_h_al = Inches(0.94)

max_pct = 40.0
for i, (label, pct, inr, body) in enumerate(alloc):
    y = al_top + row_h_al * i
    # Label
    add_text(s, Inches(0.6), y, Inches(4.4), Inches(0.3),
             label, size=12, bold=True, color=TEXT)
    add_text(s, Inches(0.6), y + Inches(0.3), Inches(4.4),
             Inches(0.5), body, size=10, color=MUTED, line_spacing=1.25)
    # bar track
    track_x = Inches(5.1)
    track_w = Inches(3.3)
    add_rect(s, track_x, y + Inches(0.12), track_w, bar_h,
             fill=BG_PANEL, corner=True)
    pct_val = float(pct.rstrip("%"))
    fill_w = int(track_w * (pct_val / max_pct))
    add_rect(s, track_x, y + Inches(0.12), fill_w, bar_h,
             fill=ACCENT, corner=True)
    add_text(s, track_x, y + Inches(0.45), track_w,
             Inches(0.3), f"{pct}  ·  {inr}", size=11, bold=True,
             color=ACCENT)

# Right column — headline numbers
hx = Inches(9.0); hy = Inches(2.0); hw = Inches(3.75); hh = Inches(4.7)
add_rect(s, hx, hy, hw, hh, fill=BG_PANEL, corner=True)
add_text(s, hx + Inches(0.3), hy + Inches(0.3), hw - Inches(0.6),
         Inches(0.35), "AT A GLANCE", size=10, bold=True, color=ACCENT)
stats = [
    ("$250K", "Pre-seed round (SAFE)"),
    ("18–24 mo", "Runway with lean core team"),
    ("10,000", "Target active users by month 12"),
    ("0.2%", "Of SOM = $36M ARR potential"),
]
sy = hy + Inches(0.85)
for big, small in stats:
    add_text(s, hx + Inches(0.3), sy, hw - Inches(0.6), Inches(0.55),
             big, size=26, bold=True, color=TEXT)
    add_text(s, hx + Inches(0.3), sy + Inches(0.55), hw - Inches(0.6),
             Inches(0.3), small, size=10, color=MUTED)
    sy += Inches(0.95)

add_footer(s, 9, TOTAL, "The Ask & Use of Funds")

# =====================================================================
# SLIDE 10 — Business model
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)
add_kicker(s, Inches(0.6), Inches(0.55), Inches(6), "Business Model")
add_text(s, Inches(0.6), Inches(0.9), Inches(12), Inches(0.8),
         "Growth first. Gated value second.",
         size=28, bold=True, color=TEXT)

# Two phase cards
bx = Inches(0.6); by = Inches(2.0); bw = Inches(6.05); bh = Inches(4.2)
add_rect(s, bx, by, bw, bh, fill=BG_PANEL, corner=True)
add_text(s, bx + Inches(0.35), by + Inches(0.25), bw - Inches(0.5),
         Inches(0.35), "PHASE 1  ·  THE GRAB", size=10, bold=True, color=ACCENT)
add_text(s, bx + Inches(0.35), by + Inches(0.65), bw - Inches(0.5),
         Inches(0.5), "100% free access", size=20, bold=True, color=TEXT)
p1_b = [
    ("Zero friction", "Onboard the first 10,000 active users at no cost."),
    ("Prove the magic", "Validate real-time biometric triggers + persistent memory in the wild."),
    ("Build the habit", "Users experience an AI that actually remembers them."),
    ("Near-zero marginal cost", "Inference runs on-device — free doesn't mean cloud-compute burn."),
]
y = by + Inches(1.55)
for t, b in p1_b:
    add_text(s, bx + Inches(0.35), y, bw - Inches(0.5), Inches(0.3),
             "•  " + t, size=12, bold=True, color=ACCENT)
    add_text(s, bx + Inches(0.55), y + Inches(0.3), bw - Inches(0.7),
             Inches(0.3), b, size=10.5, color=MUTED, line_spacing=1.25)
    y += Inches(0.62)

bx2 = Inches(6.8)
add_rect(s, bx2, by, bw, bh, fill=BG_PANEL, corner=True, line=ACCENT)
add_text(s, bx2 + Inches(0.35), by + Inches(0.25), bw - Inches(0.5),
         Inches(0.35), "PHASE 2  ·  THE FREEMIUM FLIP",
         size=10, bold=True, color=ACCENT)
add_text(s, bx2 + Inches(0.35), by + Inches(0.65), bw - Inches(0.5),
         Inches(0.5), "Anchor Premium  ·  $5/month", size=20, bold=True,
         color=TEXT)
p2_b = [
    ("Free tier", "Basic reactive AI + limited journaling. You can still text — no proactive edge."),
    ("Persistent on-device memory", "The AI retains longitudinal context of past struggles."),
    ("Proactive biometric triggers", "Active monitoring of smartwatch data; AI initiates contact during stress spikes."),
    ("Switching cost", "Months of personal context = immense stickiness. They keep their safety net."),
]
y = by + Inches(1.55)
for t, b in p2_b:
    add_text(s, bx2 + Inches(0.35), y, bw - Inches(0.5), Inches(0.3),
             "•  " + t, size=12, bold=True, color=ACCENT)
    add_text(s, bx2 + Inches(0.55), y + Inches(0.3), bw - Inches(0.7),
             Inches(0.3), b, size=10.5, color=MUTED, line_spacing=1.25)
    y += Inches(0.62)

add_text(s, Inches(0.6), Inches(6.5), Inches(12.2), Inches(0.45),
         "Economics of switching costs: once a user builds a personalized, "
         "private relationship with an AI that remembers them — conversion is inevitable.",
         size=12, color=ACCENT, align=PP_ALIGN.CENTER, font="Inter")

add_footer(s, 10, TOTAL, "Business Model")

# =====================================================================
# SLIDE 11 — The Vision / Close
# =====================================================================
s = prs.slides.add_slide(blank_layout)
add_bg(s)

# big logo backdrop on right
s.shapes.add_picture(LOGO, Inches(8.6), Inches(1.3),
                     width=Inches(4.4), height=Inches(4.4))

add_logo(s, Inches(0.6), Inches(0.55), Inches(0.55))
add_text(s, Inches(1.3), Inches(0.6), Inches(4), Inches(0.4),
         "ANCHOR", size=14, bold=True, color=TEXT)

add_kicker(s, Inches(0.6), Inches(1.55), Inches(6), "The Vision")
add_text(s, Inches(0.6), Inches(1.95), Inches(8), Inches(1.2),
         "We aren't just building for a market.\nWe are building for ourselves.",
         size=34, bold=True, color=TEXT, line_spacing=1.15)

vs = [
    ("Founder–market fit",
     "Gen Z founders in elite research, corporate, and high-stakes competitive arenas. We live the baseline stress we're solving."),
    ("Reality of high performers",
     "Stressed people don't open meditation apps. They put their heads down and push through until they break."),
    ("What Anchor is",
     "The tool we desperately needed — a proactive, private safety net that watches our backs so we don't have to."),
]
vy = Inches(3.85)
for t, b in vs:
    add_text(s, Inches(0.6), vy, Inches(7.5), Inches(0.3),
             "—  " + t, size=13, bold=True, color=ACCENT)
    add_text(s, Inches(0.85), vy + Inches(0.32), Inches(7.3), Inches(0.55),
             b, size=11, color=MUTED, line_spacing=1.3)
    vy += Inches(0.88)

# CTA strip
cta_y = Inches(6.55)
add_rect(s, Inches(0.6), cta_y, Inches(12.1), Inches(0.7),
         fill=BG_PANEL, line=ACCENT, corner=True)
add_text(s, Inches(0.85), cta_y + Inches(0.1), Inches(7), Inches(0.3),
         "The Ask", size=10, bold=True, color=ACCENT)
add_text(s, Inches(0.85), cta_y + Inches(0.32), Inches(8), Inches(0.35),
         "Raising $250,000 pre-seed on a SAFE to fund closed beta and scale to 10,000 users.",
         size=12, bold=True, color=TEXT)
add_text(s, Inches(9.0), cta_y + Inches(0.1), Inches(3.5), Inches(0.3),
         "Aryan Jain  ·  Founder / CEO", size=10, bold=True,
         color=TEXT, align=PP_ALIGN.RIGHT)
add_text(s, Inches(9.0), cta_y + Inches(0.35), Inches(3.5), Inches(0.3),
         "Tanmay Kukreja  ·  Co-Founder / CTO", size=10, bold=True,
         color=TEXT, align=PP_ALIGN.RIGHT)

add_footer(s, 11, TOTAL, "Vision & Ask")


prs.save(OUT)
print(f"Saved: {OUT}")
