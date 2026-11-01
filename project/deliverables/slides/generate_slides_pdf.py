"""Assemble the verified slide previews into a presentation PDF.

Run after generating previews with the JavaScript artifact-tool builder.
The PDF is for presenting; the PPTX retains editable text and the gesture table.
"""
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader


def main():
    folder = Path(__file__).resolve().parent
    slides = sorted((folder / 'previews').glob('slide-*.png'))
    if len(slides) != 8:
        raise SystemExit('Expected exactly eight rendered slide previews')
    output = folder / 'airboard_slides.pdf'
    canvas = Canvas(str(output), pagesize=(960, 540), pageCompression=1)
    canvas.setTitle('airboard class demonstration')
    canvas.setAuthor('airboard project')
    canvas.setSubject('AI assisted university project presentation')
    for slide in slides:
        canvas.drawImage(ImageReader(str(slide)), 0, 0, width=960, height=540)
        canvas.showPage()
    canvas.save()
    print(output)


if __name__ == '__main__':
    main()
