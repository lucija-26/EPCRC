"""Minimal flowing-document engine on top of matplotlib's PDF backend.

Produces a paginated PDF with headings, wrapped body text (with inline
mathtext), display equations, bullet lists, callout boxes and vector figures.
No LaTeX installation required.
"""

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# ----------------------------------------------------------------------------
# palette
# ----------------------------------------------------------------------------
INK = "#1a1a1a"
MUTED = "#5a5a5a"
RULE = "#c8c8c8"
ACCENT = "#1f4e79"       # deep blue: retained / structure
ACCENT2 = "#b03a2e"      # red: error / target
ACCENT3 = "#1e8449"      # green: feasible / good
ACCENT4 = "#b7791f"      # amber: caution
HULLFILL = "#dce6f1"
BOXBG = {"picture": "#eef4fb", "note": "#f2f7ef", "warn": "#fdf3e7", "key": "#f4eef8"}
BOXEDGE = {"picture": ACCENT, "note": ACCENT3, "warn": ACCENT4, "key": "#6c3483"}
BOXTITLE = {"picture": "Picture this", "note": "Implementation note",
            "warn": "Watch out", "key": "Key idea"}

# ----------------------------------------------------------------------------
# text measurement
# ----------------------------------------------------------------------------
_MDPI = 200.0
_mfig = Figure(figsize=(8.27, 11.69), dpi=_MDPI)
FigureCanvasAgg(_mfig)
_mrend = _mfig.canvas.get_renderer()
_cache = {}


def measure(s, size, weight="normal", style="normal"):
    """Width and height of a rendered string, in inches."""
    key = (s, size, weight, style)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    t = _mfig.text(0, 0, s, fontsize=size, fontweight=weight, fontstyle=style)
    try:
        bb = t.get_window_extent(_mrend)
        out = (bb.width / _MDPI, bb.height / _MDPI)
    except Exception as exc:  # surfaced loudly: a bad mathtext string
        raise ValueError("cannot render %r: %s" % (s, exc))
    finally:
        t.remove()
    _cache[key] = out
    return out


_MATH = re.compile(r"\$[^$]*\$")
_MARKUP = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")


def tokenize(s):
    """Split into (text, weight, style) tokens.

    ``**bold**`` and ``__italic__`` are honoured, and may span ``$...$``
    math; math spans are always atomic tokens.
    """
    tokens = []
    pos = 0
    for m in _MARKUP.finditer(s):
        _emit_span(tokens, s[pos:m.start()], "normal", "normal")
        if m.group(1) is not None:
            _emit_span(tokens, m.group(1), "bold", "normal")
        else:
            _emit_span(tokens, m.group(2), "normal", "italic")
        pos = m.end()
    _emit_span(tokens, s[pos:], "normal", "normal")
    return tokens


def _emit_span(tokens, chunk, weight, style):
    """Emit words from ``chunk``, keeping ``$...$`` spans as single tokens."""
    pos = 0
    for m in _MATH.finditer(chunk):
        _emit_words(tokens, chunk[pos:m.start()], weight, style)
        tokens.append((m.group(0), weight, style))
        pos = m.end()
    _emit_words(tokens, chunk[pos:], weight, style)


def _emit_words(tokens, chunk, weight, style):
    for w in chunk.split():
        tokens.append((w, weight, style))


# ----------------------------------------------------------------------------
# blocks
# ----------------------------------------------------------------------------
class Block:
    keep_together = True

    def height(self, width):
        raise NotImplementedError

    def draw(self, fig, x, y_top, width):
        raise NotImplementedError


class Para(Block):
    keep_together = False

    def __init__(self, text, size=9.3, leading=1.45, space_after=0.085,
                 indent=0.0, color=INK, align="left"):
        self.tokens = tokenize(text)
        self.size = size
        self.leading = leading * size / 72.0
        self.space_after = space_after
        self.indent = indent
        self.color = color
        self.align = align
        self._wrapped = None
        self._w = None

    def _wrap(self, width):
        if self._wrapped is not None and self._w == width:
            return self._wrapped
        sp = measure("a a", self.size)[0] - measure("aa", self.size)[0]
        lines, cur, curw = [], [], 0.0
        avail_first = width - self.indent
        for tok, wt, st in self.tokens:
            tw = measure(tok, self.size, wt, st)[0]
            avail = avail_first if not lines else width
            if cur and curw + sp + tw > avail:
                lines.append((cur, curw))
                cur, curw = [(tok, wt, st, tw)], tw
            else:
                if cur:
                    curw += sp + tw
                else:
                    curw = tw
                cur.append((tok, wt, st, tw))
        if cur:
            lines.append((cur, curw))
        self._wrapped, self._w, self._sp = lines, width, sp
        return lines

    def height(self, width):
        return len(self._wrap(width)) * self.leading + self.space_after

    def split(self, width, avail):
        """Return (head, tail) blocks so that head fits in ``avail`` inches."""
        lines = self._wrap(width)
        nfit = max(0, int((avail - 0.0) // self.leading))
        if nfit < 2 or nfit >= len(lines):
            return None
        if len(lines) - nfit < 2:
            nfit = len(lines) - 2
            if nfit < 2:
                return None
        head = _PreWrapped(lines[:nfit], self, first_indent=self.indent,
                           space_after=0.0)
        tail = _PreWrapped(lines[nfit:], self, first_indent=0.0,
                           space_after=self.space_after)
        return head, tail

    def draw(self, fig, x, y_top, width):
        self._draw_lines(fig, self._wrap(width), x, y_top, width, self.indent)

    def _draw_lines(self, fig, lines, x, y_top, width, first_indent):
        W, H = fig.get_size_inches()
        y = y_top
        for idx, (line, lw) in enumerate(lines):
            y += self.leading
            base = y - 0.24 * self.leading
            cx = x + (first_indent if idx == 0 else 0.0)
            if self.align == "center":
                cx = x + (width - lw) / 2.0
            for tok, wt, st, tw in line:
                fig.text(cx / W, 1 - base / H, tok, fontsize=self.size,
                         fontweight=wt, fontstyle=st, color=self.color,
                         ha="left", va="baseline")
                cx += tw + self._sp


class _PreWrapped(Block):
    keep_together = True

    def __init__(self, lines, parent, first_indent, space_after):
        self.lines = lines
        self.parent = parent
        self.first_indent = first_indent
        self.space_after = space_after

    def height(self, width):
        return len(self.lines) * self.parent.leading + self.space_after

    def draw(self, fig, x, y_top, width):
        self.parent._draw_lines(fig, self.lines, x, y_top, width,
                                self.first_indent)


class Heading(Block):
    def __init__(self, text, level=1, number=None):
        self.text = text
        self.level = level
        self.number = number
        self.size = {1: 15.0, 2: 11.6, 3: 9.8}[level]
        self.space_before = {1: 0.30, 2: 0.22, 3: 0.15}[level]
        self.space_after = {1: 0.13, 2: 0.09, 3: 0.06}[level]
        self.rule = level == 1

    @property
    def label(self):
        return ("%s  %s" % (self.number, self.text)) if self.number else self.text

    def height(self, width):
        h = measure(self.label, self.size, "bold")[1]
        return self.space_before + h + self.space_after + (0.07 if self.rule else 0)

    def draw(self, fig, x, y_top, width):
        W, H = fig.get_size_inches()
        h = measure(self.label, self.size, "bold")[1]
        base = y_top + self.space_before + h * 0.82
        color = ACCENT if self.level == 1 else INK
        fig.text(x / W, 1 - base / H, self.label, fontsize=self.size,
                 fontweight="bold", color=color, ha="left", va="baseline")
        if self.rule:
            yy = 1 - (base + 0.075) / H
            fig.add_artist(plt.Line2D([x / W, (x + width) / W], [yy, yy],
                                      color=ACCENT, lw=1.0,
                                      transform=fig.transFigure))


class Eq(Block):
    def __init__(self, tex, size=11.0, tag=None, pad=0.11):
        self.tex = "$" + tex + "$"
        self.size = size
        self.tag = tag
        self.pad = pad

    def height(self, width):
        return measure(self.tex, self.size)[1] + 2 * self.pad

    def draw(self, fig, x, y_top, width):
        W, H = fig.get_size_inches()
        w, h = measure(self.tex, self.size)
        cx = x + (width - w) / 2.0
        base = y_top + self.pad + h * 0.78
        fig.text(cx / W, 1 - base / H, self.tex, fontsize=self.size,
                 color=INK, ha="left", va="baseline")
        if self.tag:
            fig.text((x + width) / W, 1 - base / H, "(%s)" % self.tag,
                     fontsize=8.2, color=MUTED, ha="right", va="baseline")


class Bullet(Block):
    keep_together = True

    def __init__(self, text, marker="\u2022", size=9.3, indent=0.16,
                 space_after=0.04, color=INK):
        self.marker = marker
        self.para = Para(text, size=size, space_after=0.0, color=color)
        self.indent = indent
        self.space_after = space_after
        self.size = size

    def height(self, width):
        return self.para.height(width - self.indent) + self.space_after

    def draw(self, fig, x, y_top, width):
        W, H = fig.get_size_inches()
        base = y_top + self.para.leading - 0.24 * self.para.leading
        fig.text(x / W, 1 - base / H, self.marker, fontsize=self.size,
                 color=ACCENT, ha="left", va="baseline")
        self.para.draw(fig, x + self.indent, y_top, width - self.indent)


class Space(Block):
    def __init__(self, h=0.1):
        self.h = h

    def height(self, width):
        return self.h

    def draw(self, fig, x, y_top, width):
        pass


class Rule(Block):
    def __init__(self, pad=0.09):
        self.pad = pad

    def height(self, width):
        return 2 * self.pad

    def draw(self, fig, x, y_top, width):
        W, H = fig.get_size_inches()
        yy = 1 - (y_top + self.pad) / H
        fig.add_artist(plt.Line2D([x / W, (x + width) / W], [yy, yy],
                                  color=RULE, lw=0.7,
                                  transform=fig.transFigure))


class Callout(Block):
    def __init__(self, kind, body, title=None, size=9.0):
        self.kind = kind
        self.title = title or BOXTITLE[kind]
        self.body = [Para(b, size=size, space_after=0.05) for b in body]
        self.pad = 0.11
        self.size = size

    def height(self, width):
        iw = width - 2 * self.pad - 0.06
        h = 2 * self.pad + measure(self.title, self.size + 0.4, "bold")[1] + 0.075
        h += sum(b.height(iw) for b in self.body)
        return h + 0.14

    def draw(self, fig, x, y_top, width):
        W, H = fig.get_size_inches()
        h = self.height(width) - 0.14
        box = FancyBboxPatch(
            (x / W, 1 - (y_top + h) / H), width / W, h / H,
            boxstyle="round,pad=0,rounding_size=0.006",
            linewidth=0.9, edgecolor=BOXEDGE[self.kind],
            facecolor=BOXBG[self.kind], transform=fig.transFigure,
            zorder=0, mutation_aspect=W / H)
        fig.add_artist(box)
        th = measure(self.title, self.size + 0.4, "bold")[1]
        base = y_top + self.pad + th * 0.82
        fig.text((x + self.pad) / W, 1 - base / H, self.title,
                 fontsize=self.size + 0.4, fontweight="bold",
                 color=BOXEDGE[self.kind], ha="left", va="baseline")
        yy = y_top + self.pad + th + 0.075
        iw = width - 2 * self.pad - 0.06
        for b in self.body:
            b.draw(fig, x + self.pad, yy, iw)
            yy += b.height(iw)


class FigBlock(Block):
    def __init__(self, func, h, caption=None, num=None, args=()):
        self.func = func
        self.h = h
        self.args = args
        self.num = num
        self.caption = caption
        cap = ("**Figure %s.** " % num + caption) if caption else None
        self.cap = Para(cap, size=8.3, leading=1.32, space_after=0.0,
                        color=MUTED) if cap else None

    def height(self, width):
        h = self.h + 0.10
        if self.cap:
            h += 0.055 + self.cap.height(width)
        return h + 0.14

    def draw(self, fig, x, y_top, width):
        W, H = fig.get_size_inches()
        rect = [x / W, 1 - (y_top + 0.05 + self.h) / H, width / W, self.h / H]
        self.func(fig, rect, *self.args)
        if self.cap:
            self.cap.draw(fig, x, y_top + 0.05 + self.h + 0.055, width)


class PageBreak(Block):
    def height(self, width):
        return 0.0

    def draw(self, fig, x, y_top, width):
        pass


# ----------------------------------------------------------------------------
# document
# ----------------------------------------------------------------------------
class Doc:
    def __init__(self, pagesize=(8.27, 11.69), margin=(0.80, 0.78, 0.80, 0.72),
                 running_title=""):
        self.W, self.H = pagesize
        self.ml, self.mt, self.mr, self.mb = margin
        self.blocks = []
        self.running_title = running_title
        self.cover = None

    # -- authoring helpers ---------------------------------------------------
    def h1(self, t, n=None):
        self.blocks.append(Heading(t, 1, n))

    def h2(self, t, n=None):
        self.blocks.append(Heading(t, 2, n))

    def h3(self, t, n=None):
        self.blocks.append(Heading(t, 3, n))

    def p(self, t, **kw):
        self.blocks.append(Para(t, **kw))

    def eq(self, t, **kw):
        self.blocks.append(Eq(t, **kw))

    def li(self, t, **kw):
        self.blocks.append(Bullet(t, **kw))

    def box(self, kind, body, title=None):
        self.blocks.append(Callout(kind, body, title))

    def fig(self, func, h, caption=None, num=None, args=()):
        self.blocks.append(FigBlock(func, h, caption, num, args))

    def space(self, h=0.1):
        self.blocks.append(Space(h))

    def rule(self):
        self.blocks.append(Rule())

    def newpage(self):
        self.blocks.append(PageBreak())

    # -- rendering -----------------------------------------------------------
    @property
    def text_width(self):
        return self.W - self.ml - self.mr

    def _new_page(self, pdf, pageno):
        fig = plt.figure(figsize=(self.W, self.H))
        fig.patch.set_facecolor("white")
        return fig

    def _finish_page(self, fig, pdf, pageno):
        if pageno > 0:
            y = 1 - (self.H - self.mb + 0.28) / self.H
            fig.add_artist(plt.Line2D(
                [self.ml / self.W, (self.W - self.mr) / self.W],
                [y + 0.011, y + 0.011], color=RULE, lw=0.5,
                transform=fig.transFigure))
            fig.text(self.ml / self.W, y, self.running_title, fontsize=7.2,
                     color=MUTED, ha="left", va="top")
            fig.text((self.W - self.mr) / self.W, y, "%d" % pageno,
                     fontsize=7.6, color=MUTED, ha="right", va="top")
        pdf.savefig(fig)
        plt.close(fig)

    def render(self, path, toc_map=None):
        """Flow blocks onto pages. Returns {heading label: page number}."""
        pages_of = {}
        with PdfPages(path) as pdf:
            pageno = 0
            if self.cover is not None:
                fig = plt.figure(figsize=(self.W, self.H))
                fig.patch.set_facecolor("white")
                self.cover(fig, self)
                pdf.savefig(fig)
                plt.close(fig)

            fig = self._new_page(pdf, 1)
            pageno = 1
            y = self.mt
            bottom = self.H - self.mb
            queue = list(self.blocks)
            while queue:
                b = queue.pop(0)
                if isinstance(b, PageBreak):
                    self._finish_page(fig, pdf, pageno)
                    pageno += 1
                    fig = self._new_page(pdf, pageno)
                    y = self.mt
                    continue
                if isinstance(b, Heading):
                    pages_of[b.label] = pageno
                w = self.text_width
                h = b.height(w)
                if y + h > bottom and y > self.mt + 1e-9:
                    if isinstance(b, Para):
                        piece = b.split(w, bottom - y)
                        if piece:
                            head, tail = piece
                            head.draw(fig, self.ml, y, w)
                            queue.insert(0, tail)
                            self._finish_page(fig, pdf, pageno)
                            pageno += 1
                            fig = self._new_page(pdf, pageno)
                            y = self.mt
                            continue
                    if isinstance(b, Heading):
                        pages_of[b.label] = pageno + 1
                    self._finish_page(fig, pdf, pageno)
                    pageno += 1
                    fig = self._new_page(pdf, pageno)
                    y = self.mt
                    h = b.height(w)
                if isinstance(b, Heading) and y > self.mt + 1e-9:
                    pass
                b.draw(fig, self.ml, y, w)
                y += h
            self._finish_page(fig, pdf, pageno)
        return pages_of
