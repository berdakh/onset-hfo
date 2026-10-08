"""Sidebar glyphs, drawn rather than shipped.

A source list -- the sidebar of Mail, Finder or Notes -- puts a small
monochrome symbol beside every entry, and the symbol does most of the work of
making the list scannable: a reader finds the waveform or the document before
reading a word. Apple draws theirs from SF Symbols, which is not available on
Linux and not free to redistribute, so these are drawn with `QPainter` on a
sixteen-point grid in the palette's own colours. Simple on purpose: one or
two strokes each, the way a symbol at 16 px has to be, and the same stroke
weight throughout so the column reads as one set.

Everything is drawn at request time from the current palette, so the dark
theme gets its own set rather than a recoloured copy of the light one.
"""

from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

__all__ = ["glyph", "icon", "NAMES"]

#: The glyphs there are, by the sidebar key they are drawn beside.
NAMES = ("home", "recording", "analysis", "contacts", "map", "quality", "report",
         "assistant",
         "detectors", "outcome", "patients", "data", "architecture", "research",
         "chat")

#: Stroke weight on the sixteen-point grid; the same for every glyph.
STROKE = 1.5


def _pen(colour: QColor) -> QPen:
    pen = QPen(colour, STROKE)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _polyline(painter: QPainter, points) -> None:
    path = QPainterPath(QPointF(*points[0]))
    for x, y in points[1:]:
        path.lineTo(x, y)
    painter.drawPath(path)


def _dot(painter: QPainter, x: float, y: float, r: float, colour: QColor) -> None:
    painter.save()
    painter.setPen(Qt.NoPen)
    painter.setBrush(colour)
    painter.drawEllipse(QPointF(x, y), r, r)
    painter.restore()


def _home(p: QPainter, c: QColor) -> None:
    _polyline(p, [(2, 8.5), (8, 2.5), (14, 8.5)])
    _polyline(p, [(4, 7.5), (4, 14), (12, 14), (12, 7.5)])
    p.drawRect(QRectF(6.75, 10, 2.5, 4))


def _recording(p: QPainter, c: QColor) -> None:
    _polyline(p, [(1.5, 8), (4, 8), (5.5, 3), (7.25, 13), (9, 4.5), (10.5, 11),
                  (12, 8), (14.5, 8)])


def _contacts(p: QPainter, c: QColor) -> None:
    p.drawLine(QPointF(3, 13), QPointF(13, 3))
    for x, y in ((4.5, 11.5), (8, 8), (11.5, 4.5)):
        _dot(p, x, y, 1.9, c)


def _map(p: QPainter, c: QColor) -> None:
    p.drawRoundedRect(QRectF(2, 2, 12, 12), 2, 2)
    p.drawLine(QPointF(2, 8), QPointF(14, 8))
    p.drawLine(QPointF(8, 2), QPointF(8, 14))
    _dot(p, 11, 5, 1.6, c)


def _quality(p: QPainter, c: QColor) -> None:
    for y, knob in ((4, 5.5), (8, 10.5), (12, 7)):
        p.drawLine(QPointF(2, y), QPointF(14, y))
        _dot(p, knob, y, 1.9, c)


def _report(p: QPainter, c: QColor) -> None:
    _polyline(p, [(10, 1.5), (3.5, 1.5), (3.5, 14.5), (12.5, 14.5), (12.5, 4), (10, 1.5)])
    _polyline(p, [(10, 1.5), (10, 4), (12.5, 4)])
    for y in (7.5, 10, 12.5):
        p.drawLine(QPointF(6, y), QPointF(10, y))


def _assistant(p: QPainter, c: QColor) -> None:
    path = QPainterPath()
    path.addRoundedRect(QRectF(2, 2, 12, 9), 2.5, 2.5)
    p.drawPath(path)
    _polyline(p, [(5, 11), (5, 14), (8, 11)])
    _polyline(p, [(5.5, 6.5), (7.25, 8.25), (10.5, 4.75)])


def _detectors(p: QPainter, c: QColor) -> None:
    p.drawEllipse(QPointF(6.5, 6.5), 4.5, 4.5)
    p.drawLine(QPointF(9.9, 9.9), QPointF(14, 14))


def _outcome(p: QPainter, c: QColor) -> None:
    for x, top in ((3, 9), (7, 5), (11, 2.5)):
        p.drawRect(QRectF(x, top, 2.5, 14 - top))


def _patients(p: QPainter, c: QColor) -> None:
    p.drawEllipse(QPointF(8, 5), 2.75, 2.75)
    path = QPainterPath(QPointF(2.5, 14.5))
    path.cubicTo(2.5, 9.5, 13.5, 9.5, 13.5, 14.5)
    p.drawPath(path)


def _data(p: QPainter, c: QColor) -> None:
    p.drawEllipse(QRectF(3, 2, 10, 4))
    p.drawLine(QPointF(3, 4), QPointF(3, 12))
    p.drawLine(QPointF(13, 4), QPointF(13, 12))
    path = QPainterPath(QPointF(3, 12))
    path.cubicTo(3, 14.6, 13, 14.6, 13, 12)
    p.drawPath(path)
    path = QPainterPath(QPointF(3, 8))
    path.cubicTo(3, 10.6, 13, 10.6, 13, 8)
    p.drawPath(path)


def _architecture(p: QPainter, c: QColor) -> None:
    p.drawRect(QRectF(2, 2, 5, 4.5))
    p.drawRect(QRectF(9, 2, 5, 4.5))
    p.drawRect(QRectF(5.5, 9.5, 5, 4.5))
    p.drawLine(QPointF(4.5, 6.5), QPointF(8, 9.5))
    p.drawLine(QPointF(11.5, 6.5), QPointF(8, 9.5))


def _research(p: QPainter, c: QColor) -> None:
    path = QPainterPath(QPointF(8, 4))
    path.cubicTo(6.5, 2.5, 4, 2.5, 2, 3.5)
    path.lineTo(2, 13)
    path.cubicTo(4, 12, 6.5, 12, 8, 13.5)
    path.cubicTo(9.5, 12, 12, 12, 14, 13)
    path.lineTo(14, 3.5)
    path.cubicTo(12, 2.5, 9.5, 2.5, 8, 4)
    p.drawPath(path)
    p.drawLine(QPointF(8, 4), QPointF(8, 13.5))


def _workspace(p: QPainter, c: QColor) -> None:
    p.drawRect(QRectF(2, 3, 12, 10))
    p.drawLine(QPointF(2, 6.5), QPointF(14, 6.5))
    p.drawLine(QPointF(6.5, 6.5), QPointF(6.5, 13))
    p.drawLine(QPointF(2, 9.75), QPointF(14, 9.75))


def _analysis(p: QPainter, c: QColor) -> None:
    # Code: a pair of angle brackets around a slash.
    _polyline(p, [(5, 4), (1.5, 8), (5, 12)])
    _polyline(p, [(11, 4), (14.5, 8), (11, 12)])
    p.drawLine(QPointF(9.25, 3), QPointF(6.75, 13))


def _chat(p: QPainter, c: QColor) -> None:
    path = QPainterPath()
    path.addRoundedRect(QRectF(2, 2.5, 12, 9), 3, 3)
    p.drawPath(path)
    _polyline(p, [(4.5, 11.5), (4.5, 14.5), (8, 11.5)])
    for x in (5.5, 8, 10.5):
        _dot(p, x, 7, 0.9, c)


_DRAW = {
    "home": _home, "recording": _recording, "analysis": _analysis,
    "contacts": _contacts, "map": _map, "quality": _quality, "report": _report, "assistant": _assistant,
    "detectors": _detectors, "outcome": _outcome, "patients": _patients,
    "data": _data, "architecture": _architecture, "research": _research,
    "chat": _chat, "workspace": _workspace,
}


def glyph(name: str, colour, size: int = 16, scale: float = 2.0) -> QPixmap:
    """The glyph for `name` in `colour`, as a pixmap of `size` points.

    Drawn at `scale` times the size and marked with that device-pixel ratio,
    so it is crisp on a high-density screen and not blurred on an ordinary
    one. An unknown name draws a dot: the sidebar must never fail to build
    for want of a picture.
    """
    draw = _DRAW.get(name, lambda p, c: _dot(p, 8, 8, 2, c))
    colour = QColor(colour)
    pixmap = QPixmap(int(size * scale), int(size * scale))
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.scale(size / 16.0, size / 16.0)
    painter.setPen(_pen(colour))
    painter.setBrush(Qt.NoBrush)
    draw(painter, colour)
    painter.end()
    return pixmap


def icon(name: str, palette, size: int = 16) -> QIcon:
    """The glyph as an icon: muted at rest, in the selection's text colour
    when its row is selected, so a white symbol sits on the accent pill."""
    out = QIcon()
    out.addPixmap(glyph(name, palette.text_muted, size), QIcon.Normal)
    out.addPixmap(glyph(name, palette.text_muted, size), QIcon.Disabled)
    out.addPixmap(glyph(name, palette.accent_text, size), QIcon.Selected)
    return out
