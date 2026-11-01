"""Export the notebook's exact text and original ink to a shareable PDF."""
from pathlib import Path
from io import BytesIO
from xml.sax.saxutils import escape
import cv2


def export_notebook_pdf(notebook, destination):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    font = 'Helvetica'
    for candidate in (Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
                      Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if candidate.exists():
            if 'NotebookSans' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('NotebookSans', str(candidate)))
            font = 'NotebookSans'
            break
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normal = ParagraphStyle('NotebookBody', fontName=font, fontSize=12, leading=19, spaceAfter=9)
    title = ParagraphStyle('NotebookTitle', parent=normal, fontSize=23, leading=28, spaceAfter=12)
    note = ParagraphStyle('NotebookNote', parent=normal, fontSize=9, leading=13, textColor=colors.HexColor('#536575'))
    document = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=22*mm, rightMargin=22*mm,
                                 topMargin=20*mm, bottomMargin=20*mm,
                                 title='Airboard notebook', author='airboard project')
    story = []
    for number, page in enumerate(notebook.pages, 1):
        if number > 1:
            story.append(PageBreak())
        story.append(Paragraph(f'Airboard notebook page {number}', title))
        if page.stale:
            status = 'The handwriting changed after transcription. Compare the text with the original ink.'
        elif page.edited:
            status = 'User edited text. The original ink is retained below.'
        elif page.text:
            status = 'Unreviewed local transcription. Check the words against the original ink.'
        else:
            status = 'No transcription yet. The original ink is retained below.'
        story.append(Paragraph(status, note))
        story.append(Spacer(1, 6*mm))
        if page.text:
            # Escape markup and preserve written lines. Never summarize or add wording.
            for paragraph in page.text.split('\n\n'):
                story.append(Paragraph(escape(paragraph).replace('\n', '<br/>'), normal))
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph('Original handwriting', note))
        board = page.ink.copy()
        mask = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY) <= 10
        board[mask] = 255
        ok, encoded = cv2.imencode('.png', board)
        if not ok:
            raise OSError('Could not encode notebook ink for export')
        width = 165*mm
        height = width * notebook.height / notebook.width
        story.append(Image(BytesIO(encoded.tobytes()), width=width, height=height))
    def footer(canvas, doc):
        canvas.setFont(font, 9)
        canvas.setFillColor(colors.HexColor('#536575'))
        canvas.drawRightString(A4[0]-22*mm, 12*mm, str(doc.page))
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return destination
