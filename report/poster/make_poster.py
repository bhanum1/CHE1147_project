"""Build the A0 conference poster as an editable PowerPoint file.

This reproduces the colour-coded three-column draft as native PowerPoint
objects -- every panel is a real autoshape and every piece of copy is a real
text box, so the whole thing can be edited in PowerPoint or Keynote without
touching this script. The two result figures are the actual vector figures from
`report/figures/`, rasterized at 900 dpi (PowerPoint cannot embed PDF), which
lands near 400 dpi once they are scaled onto the A0 sheet.

Geometry is held in the draft's own coordinate system -- points on a
1263 x 1788 pt page -- and scaled up to A0 on the way out, so the numbers below
can be read straight off the draft. `SX`, `SY` and `FS` do the conversion.

Needs `python-pptx` and `pymupdf`, neither of which is a pipeline dependency:

    pip install python-pptx pymupdf

Usage (from the repo root):
    python report/poster/make_poster.py
"""
from pathlib import Path

import pymupdf
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Mm, Pt

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / 'report/figures'
OUT = Path(__file__).parent / 'EGFR_poster_A0.pptx'
CACHE = Path(__file__).parent / '.figure_cache'

# --- Scale ------------------------------------------------------------------
# The draft page, in points, and the A0 sheet it becomes.
SRC_W, SRC_H = 1263.0, 1788.0
PAGE_W_MM, PAGE_H_MM = 841.0, 1189.0
SX = PAGE_W_MM / SRC_W          # draft pt -> mm across
SY = PAGE_H_MM / SRC_H          # draft pt -> mm down
FS = (PAGE_W_MM / SRC_W) / 25.4 * 72.0   # draft pt -> real pt, for type

# Helvetica Neue ships on every Mac and is the closest available match to the
# draft's grotesque. Swap it here if Inter or Source Sans is installed.
FONT = 'Helvetica Neue'

# --- Palette ----------------------------------------------------------------
NAVY = RGBColor(0x13, 0x29, 0x3E)        # headings, masthead, dark bands
BODY = RGBColor(0x2F, 0x41, 0x55)        # body copy
MUTED = RGBColor(0x55, 0x67, 0x7B)       # captions, secondary copy
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x20, 0x6F, 0xB2)        # section 1-3 accent
TEAL = RGBColor(0x15, 0x90, 0x7B)        # section 4-5 accent
AMBER = RGBColor(0xC7, 0x82, 0x1A)       # section 6-7 accent
SKY = RGBColor(0x80, 0xC4, 0xF5)         # accent on the navy banner
TINT_BLUE = RGBColor(0xE3, 0xED, 0xF6)   # section header bands
TINT_TEAL = RGBColor(0xE0, 0xF0, 0xED)
TINT_AMBER = RGBColor(0xFB, 0xEE, 0xDD)
PANEL = RGBColor(0xF5, 0xF6, 0xF8)       # neutral panels
PANEL_BLUE = RGBColor(0xEF, 0xF4, 0xFA)  # emphasised panels
PANEL_AMBER = RGBColor(0xFD, 0xF3, 0xE7)
RULE = RGBColor(0xE6, 0xE9, 0xEE)        # hairlines inside the table
DASH = RGBColor(0x9E, 0xB3, 0xC6)        # the train/test divider

# --- Grid -------------------------------------------------------------------
COL = [50.0, 448.0, 846.0]   # column left edges
COL_W = 366.0
BAR = 4.0                    # width of a section's accent bar

# Helvetica Neue sets a little wider than the font the draft was made with, so
# body copy and captions are stepped down from the draft's sizes to keep every
# block inside the panel it belongs to. Headings are unchanged.
BODY_SZ = 14.8
CAP_SZ = 12.0


def x(pt):
    return Mm(pt * SX)


def y(pt):
    return Mm(pt * SY)


def fs(pt):
    """Draft point size -> real point size on the A0 sheet."""
    return Pt(round(pt * FS, 1))


def tracking(run, hundredths):
    """Letter-spacing, which python-pptx does not expose."""
    run.font._rPr.set('spc', str(int(hundredths)))


def rect(slide, left, top, width, height, fill=None,
         line=None, line_w=1.0):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                   x(left), y(top), x(width), y(height))
    shape.shadow.inherit = False
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = fs(line_w)
    shape.text_frame.text = ''
    return shape


def dashed_line(slide, x0, y0, x1, y1, colour, width_pt):
    """A dashed rule. Drawn as a connector so only the line itself is dashed."""
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                      x(x0), y(y0), x(x1), y(y1))
    line.line.color.rgb = colour
    line.line.width = fs(width_pt)
    ln = line.line._get_or_add_ln()
    ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    return line


def textbox(slide, left, top, width, height, paras, anchor=MSO_ANCHOR.TOP,
            align=PP_ALIGN.LEFT):
    """Lay out one text box.

    `paras` is a list of paragraphs; each is a list of runs, and each run is
    `(text, size, colour, bold)` with optional trailing keyword-ish extras
    handled by `_run`. Paragraph-level spacing rides on the first run's dict.
    """
    box = slide.shapes.add_textbox(x(left), y(top), x(width), y(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    for i, spec in enumerate(paras):
        runs = spec['runs']
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = spec.get('align', align)
        if 'leading' in spec:
            p.line_spacing = spec['leading']
        if 'after' in spec:
            p.space_after = fs(spec['after'])
        for r in runs:
            run = p.add_run()
            run.text = r['t']
            run.font.name = FONT
            run.font.size = fs(r.get('sz', 15.5))
            run.font.bold = r.get('b', False)
            run.font.italic = r.get('i', False)
            run.font.color.rgb = r.get('c', BODY)
            if r.get('spc'):
                tracking(run, r['spc'])
    return box


def para(runs, leading=1.30, after=0.0, align=None):
    spec = {'runs': runs, 'leading': leading, 'after': after}
    if align is not None:
        spec['align'] = align
    return spec


def t(text, sz=BODY_SZ, c=BODY, b=False, i=False, spc=None):
    return {'t': text, 'sz': sz, 'c': c, 'b': b, 'i': i, 'spc': spc}


def section(slide, col, top, label, tint, accent, height=43.0):
    """A numbered section header: tinted band, accent bar, navy title."""
    rect(slide, COL[col], top, COL_W, height, fill=tint)
    rect(slide, COL[col], top, BAR, height, fill=accent)
    textbox(slide, COL[col] + 20, top, COL_W - 28, height,
            [para([t(label, sz=22.6, c=NAVY, b=True)], leading=1.0)],
            anchor=MSO_ANCHOR.MIDDLE)


def figure(slide, pdf_name, left, top, width):
    """Place a repo figure, rasterized from its vector PDF at 900 dpi."""
    CACHE.mkdir(exist_ok=True)
    png = CACHE / f'{pdf_name}_900.png'
    doc = pymupdf.open(FIGURES / f'{pdf_name}.pdf')
    rectangle = doc[0].rect
    if not png.exists():
        doc[0].get_pixmap(dpi=900, alpha=False).save(png)
    height = width * rectangle.height / rectangle.width
    slide.shapes.add_picture(str(png), x(left), y(top), x(width), y(height))


def build():
    prs = Presentation()
    prs.slide_width = Mm(PAGE_W_MM)
    prs.slide_height = Mm(PAGE_H_MM)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # ======================= Masthead =======================================
    textbox(slide, 50, 42, 800, 20,
            [para([t('CHE1147 · CHEMICAL DATA SCIENCE AND ENGINEERING · FALL 2026',
                     sz=14.6, c=BLUE, b=True, spc=120)], leading=1.0)])
    textbox(slide, 50, 60, 840, 96,
            [para([t('Predicting EGFR inhibition with machine learning on '
                     'standard molecular features', sz=36.5, c=NAVY, b=True)],
                  leading=1.12)])
    textbox(slide, 900, 66, 314, 78,
            [para([t('Bhanu Mamillapalli', sz=17.2, c=NAVY, b=True)], leading=1.38),
             para([t('Zhijian Zhuo', sz=17.2, c=NAVY, b=True)], leading=1.38),
             para([t('Ruilin Zhang', sz=17.2, c=NAVY, b=True)], leading=1.38)],
            align=PP_ALIGN.RIGHT)
    textbox(slide, 900, 155, 314, 16,
            [para([t('[YOUR DEPARTMENT / AFFILIATION]', sz=12.0, c=MUTED, spc=80)],
                  leading=1.0)], align=PP_ALIGN.RIGHT)

    rect(slide, 50, 181, 1162, 3.8, fill=NAVY)

    # ======================= Key-result banner ==============================
    rect(slide, 50, 200, 1162, 83, fill=NAVY)
    textbox(slide, 74, 200, 1114, 83,
            [para([t('A random forest on Morgan fingerprints and RDKit descriptors reaches ',
                     sz=20.3, c=WHITE),
                   t('0.874 ROC-AUC', sz=20.3, c=SKY, b=True),
                   t(' on molecules whose scaffolds it has never seen — against a '
                     '0.500 majority baseline.', sz=20.3, c=WHITE)], leading=1.18)],
            anchor=MSO_ANCHOR.MIDDLE)

    # ======================= Column 1 =======================================
    section(slide, 0, 312, '1 · The problem', TINT_BLUE, BLUE)
    textbox(slide, 50, 368, 366, 244, [
        para([t('Mutated protein kinases send excess growth signals and drive tumour '
                'formation. EGFR is among the most important kinase targets in cancer.')],
             after=16),
        para([t('Experimental screening of candidate inhibitors is slow and expensive, '
                'so computational triage is valuable.')], after=16),
        para([t('Prior work shows the molecular representation and the splitting '
                'strategy matter as much as the algorithm does.')]),
    ])

    rect(slide, 50, 616, 366, 120, fill=PANEL_BLUE)
    rect(slide, 50, 616, BAR, 120, fill=BLUE)
    textbox(slide, 73, 626, 320, 100, [
        para([t('RESEARCH QUESTION', sz=14.6, c=BLUE, b=True, spc=120)], after=11),
        para([t('Can a model predict whether a molecule inhibits EGFR — and what '
                'accuracy can it reach on chemistry it has never seen?',
                sz=16.4, c=NAVY, b=True)], leading=1.26),
    ])

    section(slide, 0, 764, '2 · Dataset', TINT_BLUE, BLUE, height=44)
    for left, number, caption in ((50.0, '4,982', 'molecules'),
                                  (175.8, '69.2%', 'inhibitors'),
                                  (300.0, '2,265', 'raw features')):
        rect(slide, left, 824, 116, 70, fill=PANEL)
        textbox(slide, left, 824, 116, 70, [
            para([t(number, sz=24.2, c=BLUE, b=True)], leading=1.0, after=6,
                 align=PP_ALIGN.CENTER),
            para([t(caption, sz=CAP_SZ, c=MUTED)], leading=1.0, align=PP_ALIGN.CENTER),
        ], anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, 50, 906, 366, 112, [
        para([t('EGFR subset of the Kaggle Cancer Inhibitors dataset, labelled from '
                'reported IC50 — inhibitor if IC50 < 10 µM. Duplicate measurements '
                'collapsed; molecules without a usable structure or label dropped.')]),
    ])

    section(slide, 0, 1044, '3 · Why scaffold splitting', TINT_BLUE, BLUE, height=44)
    rect(slide, 50, 1108, 154, 116, fill=PANEL_BLUE, line=BLUE, line_w=1.4)
    rect(slide, 258, 1108, 154, 116, fill=PANEL_AMBER, line=AMBER, line_w=1.4)
    dashed_line(slide, 231, 1108, 231, 1224, DASH, 1.6)
    textbox(slide, 63, 1112, 132, 104, [
        para([t('TRAIN · 3,986', sz=15.0, c=BLUE, b=True)], leading=1.0, after=9),
        para([t('Most common Murcko scaffolds, added until 80% of molecules are held',
                sz=CAP_SZ, c=MUTED)], leading=1.3),
    ])
    textbox(slide, 270, 1112, 132, 104, [
        para([t('TEST · 996', sz=15.0, c=AMBER, b=True)], leading=1.0, after=9),
        para([t('Remaining scaffolds — no core shared with any training molecule',
                sz=CAP_SZ, c=MUTED)], leading=1.3),
    ])
    textbox(slide, 50, 1238, 372, 70, [
        para([t('A random split lets near-duplicates of training molecules leak into '
                'the test set and inflates the score. Holding out whole Bemis–Murcko '
                'scaffolds makes the test score an estimate of performance on new '
                'chemistry.', sz=CAP_SZ, c=MUTED)], leading=1.32),
    ])

    # ======================= Column 2 =======================================
    section(slide, 1, 312, '4 · The pipeline', TINT_TEAL, TEAL)
    steps = [
        (372.0, 72.0, 'Curate',
         'Deduplicate, drop unusable structures and missing labels → 4,982 molecules', False),
        (456.0, 72.0, 'Featurize — RDKit',
         '2,048-bit Morgan fingerprint (radius 2) + 217 two-dimensional descriptors', False),
        (540.0, 56.0, 'Split — Murcko scaffold 80/20',
         'No scaffold shared between train and test', True),
        (608.0, 88.0, 'Preprocess',
         'Drop missing, zero-variance and |r| > 0.9 features; log-scale kurtosis > 7; '
         'standardize — every step fitted on training data only', False),
        (708.0, 56.0, 'Select — 5-fold scaffold CV',
         '51 hyperparameter configurations ranked by ROC-AUC', False),
        (776.0, 72.0, 'Test — once',
         'Refit on all training molecules, score on the held-out scaffolds', False),
    ]
    for top, height, title, detail, highlight in steps:
        rect(slide, COL[1], top, COL_W, height,
             fill=PANEL_BLUE if highlight else PANEL)
        rect(slide, COL[1], top, BAR, height, fill=BLUE if highlight else NAVY)
        textbox(slide, 466, top + 8, 334, height - 14, [
            para([t(title, sz=15.9, c=BLUE if highlight else NAVY, b=True)],
                 leading=1.0, after=7),
            para([t(detail, sz=CAP_SZ, c=MUTED)], leading=1.3),
        ])
    textbox(slide, 448, 860, 366, 40, [
        para([t('2,265 → ', sz=CAP_SZ, c=MUTED),
              t('2,165 features retained', sz=CAP_SZ, c=NAVY, b=True),
              t(' (2,007 fingerprint bits + 158 descriptors).', sz=CAP_SZ, c=MUTED)],
             leading=1.32),
    ])

    section(slide, 1, 924, '5 · Four models, four questions', TINT_TEAL, TEAL)
    models = [
        (978.0, 'Majority class',
         'What do the metrics look like with no model at all? · always predicts "inhibitor"'),
        (1055.0, 'Logistic regression',
         'Is the feature space close to linearly separable? · L2, C = 0.1'),
        (1115.0, 'Tanimoto k-NN',
         'Is "similar molecule → similar activity" enough? · k = 21, distance weighted'),
        (1192.0, 'Random forest',
         'Do nonlinear interactions between bits and descriptors help? · unlimited depth, '
         '1 sample per leaf, √p features'),
    ]
    for top, name, detail in models:
        textbox(slide, 448, top, 366, 72, [
            para([t(name, sz=16.7, c=NAVY, b=True)], leading=1.0, after=7),
            para([t(detail, sz=CAP_SZ, c=MUTED)], leading=1.32),
        ])

    # ======================= Column 3 =======================================
    section(slide, 2, 312, '6 · Test performance', TINT_AMBER, AMBER)
    figure(slide, 'figure2_roc_scaffold', 876.9, 376.0, 304.2)
    results_table(slide)
    textbox(slide, 846, 866, 356, 60, [
        para([t('996 held-out molecules, none sharing a scaffold with the training set. '
                'All three learned models finish within 0.022 ROC-AUC of each other.',
                sz=CAP_SZ, c=MUTED)], leading=1.32),
    ])

    section(slide, 2, 944, '7 · Where it fails', TINT_AMBER, AMBER, height=45)
    figure(slide, 'figure3_confusion_random_forest', 908.0, 1012.0, 242.0)
    textbox(slide, 846, 1273, 356, 80, [
        para([t('Recall on inhibitors is high (636 of 679), but 173 of 317 '
                'non-inhibitors are called inhibitors at the default 0.5 threshold. In a '
                'screening workflow a false positive costs one assay; the threshold can '
                'be moved, and ROC-AUC is threshold-free.', sz=CAP_SZ, c=MUTED)],
             leading=1.32),
    ])

    # ======================= Footer =========================================
    rect(slide, 50, 1475, 1162, 4, fill=NAVY)
    footer = [
        (0, 'Conclusions', [
            [t('Machine learning on standard molecular features predicts EGFR '
               'inhibition on unseen scaffolds: '),
             t('0.874 vs 0.500', b=True), t(' for the baseline.')],
            [t('A tuned random forest beats a simple Tanimoto k-NN by only '
               '0.015 ROC-AUC.')],
            [t('A model trained on published EGFR data can prioritize compounds from '
               'chemistry outside the series it saw.')],
        ]),
        (1, 'Limitations', [
            [t('A fixed 10 µM cut discards potency; a borderline compound is treated '
               'like a highly potent one.')],
            [t('Inactive compounds are under-reported, so the 69% class balance '
               'reflects publication practice, not chemical space.')],
            [t('The representation was fixed, not ablated — the fingerprint and '
               'descriptor contributions are unmeasured.')],
        ]),
        (2, 'Future work', [
            [t('Train across the other kinases in the same source so one model learns '
               'shared patterns.')],
            [t('Predict IC50 as a regression target instead of applying a hard '
               'threshold.')],
        ]),
    ]
    for col, title, blocks in footer:
        rect(slide, COL[col], 1498, COL_W, 43, fill=NAVY)
        textbox(slide, COL[col] + 12, 1498, COL_W - 20, 43,
                [para([t(title, sz=22.0, c=WHITE, b=True)], leading=1.0)],
                anchor=MSO_ANCHOR.MIDDLE)
        textbox(slide, COL[col], 1552, COL_W, 180,
                [para(runs, after=16) for runs in blocks])

    textbox(slide, 846, 1670, 366, 80, [
        para([t('Code and environment: ', sz=CAP_SZ, c=MUTED),
              t('[YOUR GITHUB REPO URL]', sz=CAP_SZ, c=NAVY, b=True)],
             leading=1.3, after=4),
        para([t('Key references: Rogers & Hahn, ', sz=CAP_SZ, c=MUTED),
              t('JCIM', sz=CAP_SZ, c=MUTED, i=True),
              t(' 2010 · Bemis & Murcko, ', sz=CAP_SZ, c=MUTED),
              t('J. Med. Chem.', sz=CAP_SZ, c=MUTED, i=True),
              t(' 1996 · Wu et al., ', sz=CAP_SZ, c=MUTED),
              t('Chem. Sci.', sz=CAP_SZ, c=MUTED, i=True),
              t(' 2018 · Zhang & Li, ', sz=CAP_SZ, c=MUTED),
              t('New J. Chem.', sz=CAP_SZ, c=MUTED, i=True),
              t(' 2023 · Kaggle Cancer Inhibitors (CC BY-NC-SA 4.0)', sz=CAP_SZ, c=MUTED)],
             leading=1.3),
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f'wrote {OUT}')


def cell_border(cell, edge, colour, width_pt):
    """Set one border on a table cell. python-pptx has no API for this."""
    tc = cell._tc.get_or_add_tcPr()
    tag = {'top': 'a:lnT', 'bottom': 'a:lnB', 'left': 'a:lnL', 'right': 'a:lnR'}[edge]
    for existing in tc.findall(qn(tag)):
        tc.remove(existing)
    ln = tc.makeelement(qn(tag), {'w': str(int(width_pt * 12700)), 'cap': 'flat'})
    fill = tc.makeelement(qn('a:solidFill'), {})
    fill.append(tc.makeelement(qn('a:srgbClr'), {'val': f'{colour.__str__()}'}))
    ln.append(fill)
    # Borders must appear in schema order: L, R, T, B.
    order = ['a:lnL', 'a:lnR', 'a:lnT', 'a:lnB']
    tc.insert(min(len(tc), sum(1 for e in tc if e.tag in
                               [qn(o) for o in order[:order.index(tag)]])), ln)


def results_table(slide):
    """Section 6's results table, as a real PowerPoint table."""
    rows = [
        ('Model', 'CV ROC-AUC', 'Test', 'F1'),
        ('Random forest', '0.880 ± 0.021', '0.874', '0.86'),
        ('KNN (Tanimoto)', '0.850 ± 0.025', '0.859', '0.85'),
        ('Logistic regression', '0.854 ± 0.020', '0.852', '0.85'),
        ('Majority baseline', '0.500', '0.500', '0.81'),
    ]
    widths = [148.0, 112.0, 56.0, 50.0]
    top, header_h, row_h = 700.0, 32.0, 32.0

    graphic = slide.shapes.add_table(len(rows), 4, x(COL[2]), y(top),
                                     x(sum(widths)), y(header_h + 4 * row_h))
    table = graphic.table
    # Turn off the built-in style so the fills below are what shows.
    table.first_row = False
    table.horz_banding = False
    for i, w in enumerate(widths):
        table.columns[i].width = x(w)
    table.rows[0].height = y(header_h)
    for r in range(1, 5):
        table.rows[r].height = y(row_h)

    for r, record in enumerate(rows):
        header = r == 0
        muted = r == 4
        for c, value in enumerate(record):
            cell = table.cell(r, c)
            cell.margin_left = cell.margin_right = x(2)
            cell.margin_top = cell.margin_bottom = 0
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL_BLUE if r == 1 else WHITE

            tf = cell.text_frame
            tf.word_wrap = False
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = value
            run.font.name = FONT
            run.font.size = fs(13.6)
            run.font.bold = header or r == 1
            if muted:
                run.font.color.rgb = MUTED
            elif r == 1 and c == 2:
                run.font.color.rgb = BLUE     # the headline number
            else:
                run.font.color.rgb = NAVY

            if header:
                cell_border(cell, 'bottom', NAVY, 2.0 * FS)
            elif r in (2, 3, 4):
                cell_border(cell, 'top', RULE, 1.0 * FS)


if __name__ == '__main__':
    build()
