"""Build the concise project report from recorded evidence, without invented results.

Run with a Python environment containing reportlab. Pass --evidence to include
confirmed software-test counts. No camera or network access is used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
LINK = 'https://github.com/Szobo8Nunez9/airboard'
DARK = colors.HexColor('#152C3A')
TEAL = colors.HexColor('#087C83')
MUTED = colors.HexColor('#596873')
LIGHT = colors.HexColor('#EDF3F4')

# Embed familiar fonts when available; keep a portable built-in fallback.
FONT, BOLD, MONO = 'Helvetica', 'Helvetica-Bold', 'Courier'
font_dir = Path('/System/Library/Fonts/Supplemental')
if all((font_dir/name).is_file() for name in ('Arial.ttf', 'Arial Bold.ttf', 'Courier New.ttf')):
    pdfmetrics.registerFont(TTFont('AirboardSans', str(font_dir/'Arial.ttf')))
    pdfmetrics.registerFont(TTFont('AirboardSansBold', str(font_dir/'Arial Bold.ttf')))
    pdfmetrics.registerFont(TTFont('AirboardMono', str(font_dir/'Courier New.ttf')))
    pdfmetrics.registerFontFamily('AirboardSans', normal='AirboardSans', bold='AirboardSansBold')
    FONT, BOLD, MONO = 'AirboardSans', 'AirboardSansBold', 'AirboardMono'


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def content(evidence):
    tests = evidence.get('automated_tests', {})
    passed = tests.get('passed')
    failed = tests.get('failed')
    test_result = (f'{passed} passed, {failed} failed' if isinstance(passed, int) and isinstance(failed, int)
                   else 'A final run is not yet recorded')
    repair_status = evidence.get('repair_status', 'The repair is being checked with targeted regression tests.').rstrip('. ') + '.'
    # Synthetic metrics are intentionally not promoted to camera or AI accuracy.
    return {
        'test_result': test_result,
        'test_command': tests.get('command', 'python -m pytest tests/ -q'),
        'camera_status': evidence.get('camera_status', 'Not measured in this verification run').rstrip('. '),
        'live_ai_status': evidence.get('live_ai_status', 'Not measured in this verification run').rstrip('. '),
        'repair_status': repair_status,
    }


def build_pdf(output, facts):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportTitle', fontName=BOLD, fontSize=30,
                              leading=34, textColor=colors.black, spaceAfter=12))
    styles.add(ParagraphStyle('SectionTitle', fontName=BOLD, fontSize=20,
                              leading=24, textColor=colors.black, spaceAfter=13))
    styles.add(ParagraphStyle('SmallHeading', fontName=BOLD, fontSize=12,
                              leading=16, textColor=DARK, spaceBefore=11, spaceAfter=5))
    styles.add(ParagraphStyle('BodyCopy', fontName=FONT, fontSize=10.5, leading=15,
                              spaceAfter=8, textColor=DARK, alignment=TA_LEFT))
    styles.add(ParagraphStyle('CaptionCopy', fontName=FONT, fontSize=9, leading=12,
                              spaceAfter=8, textColor=MUTED))
    styles.add(ParagraphStyle('TableCopy', fontName=FONT, fontSize=9.5, leading=13,
                              textColor=DARK))
    styles.add(ParagraphStyle('TableHead', parent=styles['TableCopy'], fontName=BOLD,
                              textColor=colors.white))
    styles.add(ParagraphStyle('CodeCopy', fontName=MONO, fontSize=9.5, leading=13,
                              spaceAfter=7, textColor=DARK))
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=22*mm,
                            rightMargin=22*mm, topMargin=22*mm, bottomMargin=22*mm,
                            title='Airboard project report', author='Airboard project',
                            subject='AI assisted webcam whiteboard prototype')
    story = []
    def p(text, style='BodyCopy'):
        story.append(Paragraph(text, styles[style]))
    def heading(text):
        p(text, 'SmallHeading')
    def table(rows, widths, header=True):
        cells = [[Paragraph(str(c), styles['TableHead' if header and r == 0 else 'TableCopy'])
                  for c in row] for r, row in enumerate(rows)]
        t = Table(cells, colWidths=widths, hAlign='LEFT')
        commands = [('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LEFTPADDING', (0,0), (-1,-1), 9), ('RIGHTPADDING', (0,0), (-1,-1), 9),
                    ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                    ('LINEBELOW', (0,0), (-1,0), .7, TEAL)]
        if header:
            commands += [('BACKGROUND', (0,0), (-1,0), DARK),
                         ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT])]
        t.setStyle(TableStyle(commands))
        story.extend([t, Spacer(1, 8)])
    # Page 1
    p('Airboard project report', 'ReportTitle')
    p('An AI assisted webcam whiteboard', 'SectionTitle')
    p('Submission deadline: 2 November 2026 at 08:00, Asia/Taipei', 'CaptionCopy')
    p('Airboard is a Python prototype for drawing on a screen with hand movements. '
      'A webcam observes the hand, simple gestures choose a tool, and a separate canvas keeps the drawing. '
      'The goal is a small idea that another student can try and understand, with optional AI help for interpreting a saved board.')
    heading('What I made')
    p('The program supports drawing, moving without drawing, erasing, and clearing. '
      'It can show the canvas over the camera image or on a white background, change ink colors, and save a PNG. '
      'A mouse mode provides a way to try the canvas and demonstrate the interface when a webcam is unavailable.')
    table([
        ['Hand gesture', 'Action'],
        ['Index finger extended', 'Draw a stroke'],
        ['Index and middle fingers extended', 'Erase between the fingertips'],
        ['Open hand', 'Hover without adding ink'],
        ['Fist held for one second', 'Clear the board once'],
    ], [doc.width*.48, doc.width*.52])
    heading('Who it is for')
    p('The intended users are students explaining a quick equation or sketch in a study group or classroom. '
      'They can experiment with a webcam instead of buying a drawing tablet. This is a prototype for exploration. '
      'It has not been evaluated as an accessibility aid or a replacement for precise handwriting tools.')
    p('Project repository: <link href="'+LINK+'" color="#087C83">'+LINK+'</link>', 'CaptionCopy')
    p('The repository contains the source code, software checks, project report, and presentation slides.', 'CaptionCopy')
    story.append(PageBreak())
    # Page 2
    p('How the prototype works', 'SectionTitle')
    heading('Camera and drawing pipeline')
    p('OpenCV captures and mirrors the camera image. MediaPipe Hands supplies 21 hand landmarks. '
      'The program converts those landmarks to screen positions and checks which fingers appear extended. '
      'The index fingertip becomes the drawing point. Ink stays in a separate image buffer, so displaying or erasing a stroke does not alter the camera frame.')
    p('The fingertip position uses an exponential moving average. At alpha 0.5, the new position receives half the weight and the previous smoothed position receives half. '
      'Alpha 1.0 follows the raw coordinates. The filter resets when a stroke ends, so a new stroke does not connect to an old position.')
    p('Fixed EMA is the default. An optional adaptive mode varies the smoothing gain during motion. The recorded alpha comparison uses fixed EMA.')
    p('p[t] = alpha * raw[t] + (1 - alpha) * p[t-1]', 'CodeCopy')
    p('A fist starts a one-second hold timer. Losing the detected hand interrupts the hold. '
      'Holding the fist should produce one clear event, and opening the hand prepares the timer for a future hold. '
      'The gesture rules assume a reasonably upright, visible hand. Rotation and occlusion remain limitations.')
    heading('Optional AI interpretation')
    p('Pressing A exports a white-background PNG of the board. Separate provider adapters send that same image to Claude and an OpenAI vision model in parallel. '
      'Each response carries the provider, model, elapsed time, and success or error state. The interface keeps the two replies separate.')
    p('The providers are optional. Real requests require API keys in a local .env file and may use paid API services. '
      'Without keys, the canvas still works and the program explains why interpretation is unavailable. Explicit demo mode shows interface examples and says that no model interpreted the board.')
    p('The comparison checks explicit final-answer fields rather than numbers found anywhere in an explanation. '
      'A match means the displayed final answers match under a simple normalization rule. It does not prove correctness. '
      'Unclear or missing replies produce an unassessed comparison.')
    heading('How I used AI')
    p('Antigravity supplied the initial implementation, sample boards, tests, and draft report and slides. '
      'ChatGPT independently inspected the source and reproduced edge cases, then helped repair the program and rewrite the supporting material. '
      'The work involved reading generated code, running checks, and correcting claims instead of accepting every generated statement.')
    p('Claude was part of the intended architecture and review plan. The available project evidence does not independently verify the earlier review actions attributed to Claude. '
      'Those actions are not counted as completed contributions in this report.')
    story.append(PageBreak())
    # Page 3
    p('What I checked', 'SectionTitle')
    p('The checks separate software behavior from performance in the real world. '
      'A synthetic coordinate test can check a gesture rule, but it cannot measure a webcam under bright or dim lighting. '
      'A demo answer can check a panel, but it cannot measure a model\'s ability to read handwriting.')
    table([
        ['Check', 'Recorded evidence and limit'],
        ['Automated regression suite', escape(facts['test_result']) + '. Tests use controlled inputs and blocked network calls.'],
        ['Gesture simulation', 'Generated landmark trials check the classifier. They are labeled synthetic and preserve IDLE or missing detections.'],
        ['Smoothing simulation', '360 generated points across three paths, replayed at alpha 1.0 and 0.5. The jitter metric decreased, while geometric error increased. This does not measure user handwriting.'],
        ['Webcam field trials', escape(facts['camera_status']) + '. No camera-accuracy percentage is claimed.'],
        ['Live AI board evaluation', escape(facts['live_ai_status']) + '. No live-model accuracy or response-time result is claimed.'],
    ], [doc.width*.31, doc.width*.69])
    heading('Problems found in the first draft')
    p('The independent review reproduced six errors. Failed provider calls became prepared correct answers in the benchmark. '
      'The agreement heuristic accepted shared equation numbers even when final answers differed. The grader accepted a correct number mentioned as a rejected candidate. '
      'An IDLE prediction became HOVER and earned credit. The clear timer survived lost tracking. Provider errors produced blank answer panels.')
    heading('What the repair changes')
    p('The revised flow keeps failures as failures and distinguishes live, demo, and unavailable results. '
      'It compares explicit final answers, leaves unassessable comparisons unknown, and uses a fixed human rubric for a real 15-board study. '
      'Tracking loss resets the gesture state, and each provider panel shows its own failure message. '
      'Synthetic test outputs remain clearly labeled.')
    p(escape(facts['repair_status']))
    heading('Reproducing the software checks')
    p(escape(facts['test_command']), 'CodeCopy')
    p('The repository keeps test source and generated evaluation evidence so another reader can inspect what each check did. '
      'The former 100% accuracy claims and unverified review-credit entries are not used as evidence in this report.')
    story.append(PageBreak())
    # Page 4
    p('Limitations and next steps', 'SectionTitle')
    heading('What remains uncertain')
    p('The user confirmed that the Terminal launch worked on 1 November. Gesture comfort, drawing stability, and performance in the presentation room still need a full rehearsal. '
      'Lighting, camera distance, hand angle, and occluded fingers can affect the landmarks. The one-second clear gesture also needs a practical check with a person, even when its timer passes a software test. '
      'A clean installation and the camera setup still need validation on the presentation computer.')
    p('Real AI interpretation has not been measured in this verification run. An API may reject a key or model, time out, misread a stroke, or produce a confident wrong solution. '
      'Exact final-answer matching can also mark equivalent mathematical notation as different. The user must review both replies against the board.')
    heading('Next experiments')
    p('First, collect 25 observed trials for each gesture and record the actual lighting and camera distance for every trial. '
      'Keep missed detections in the data. Next, replay the same captured fingertip sequence at alpha 1.0 and 0.5 to compare jitter, lag, and corner shape. '
      'Finally, submit the same 15 board images to both live providers and retain full replies, model identifiers, errors, and timing. '
      'Grade the answers against a fixed rubric and count matching wrong answers separately from matching correct answers.')
    heading('Notebook and local text recognition')
    p('Notebook mode keeps multiple pages with the original ink beside editable text. Apple Vision reads the page locally. Formatting tidies spacing without rewriting the words. The user reviews uncertain recognition, can edit or undo text changes, and exports the notebook as a PDF. A generated printed-text check passed. Real handwriting accuracy remains unmeasured.')
    heading('Class demonstration')
    p('The planned demonstration starts with one short stroke, then hover, erase, and clear. '
      'It switches to the white background and saves a board image, then shows a notebook page and its editable text. Mouse mode is available to show the idea if camera access fails. '
      'The AI panels are shown as live only when actual requests succeed. Otherwise the presenter uses the labeled demo and explains the remaining work.')
    heading('Sources and implementation references')
    references = [
        ('MediaPipe Hands legacy solution', 'https://chuoling.github.io/mediapipe/solutions/hands.html'),
        ('OpenAI images and vision', 'https://developers.openai.com/api/docs/guides/images-vision?api-mode=chat'),
        ('Anthropic vision input guide', 'https://platform.claude.com/docs/en/build-with-claude/vision'),
        ('OpenCV image encoding', 'https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html'),
    ]
    for title, url in references:
        p('<link href="'+escape(url)+'" color="#087C83">'+escape(title)+'</link><br/>'+escape(url), 'CaptionCopy')
    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(22*mm, A4[1]-13*mm, 'AIRBOARD')
        canvas.drawRightString(A4[0]-22*mm, 13*mm, str(doc.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=page, onLaterPages=page)
    return output


def markdown_report(facts):
    return f'''# Airboard project report

An AI assisted webcam whiteboard. Prepared for the class presentation on 2 November 2026.

## What I made

Airboard is a Python prototype for drawing on a screen with hand movements. OpenCV captures a webcam image, MediaPipe Hands supplies 21 landmarks, and simple rules map finger combinations to drawing tools. Ink stays on a separate canvas that can appear over the video or on a white background. The program can change ink colors and save PNG images. Mouse mode lets another student try the canvas when a webcam is unavailable.

Index finger extended means draw. Index and middle extended means erase. An open hand hovers. A fist held for one second clears the canvas once. An exponential moving average smooths the fingertip. Alpha 0.5 blends the new point with the prior smoothed point, while alpha 1.0 follows raw points. Ending a stroke resets the filter. Fixed EMA is the default. An optional adaptive mode varies the gain during motion; the recorded alpha comparison uses fixed EMA.

## Notebook

Notebook mode keeps multiple pages and original ink beside editable text. Apple Vision recognizes text locally. Spacing cleanup preserves recognized wording. Users review and correct words before exporting a PDF. A generated printed-text check passed. Real handwriting accuracy remains unmeasured.

## Who it is for

The intended users are students explaining quick equations or sketches to classmates or a study group. It is a small experiment with a normal webcam, rather than a validated replacement for a drawing tablet or an accessibility aid.

## How I made it with AI

Antigravity supplied the initial implementation, sample boards, tests, and draft artifacts. ChatGPT independently reviewed the source, reproduced errors, and helped repair the program and rewrite its supporting material. Claude was part of the intended architecture and review plan, but the earlier review actions attributed to Claude were not independently verified and are not counted as completed work here.

The optional runtime exports the same board PNG to Claude and an OpenAI vision model in parallel. Each provider has its own result and error state. Real requests require keys in a local .env file. Missing keys leave interpretation unavailable rather than producing a fictional answer. Explicit demo mode labels its panel examples and makes no model request.

A comparison uses explicit final-answer fields. Matching answers can both be wrong, and mathematically equivalent notation can compare unequal. Missing or uncertain responses produce an unassessed result.

## What I checked

- Automated regression suite: {facts['test_result']}. Command: `{facts['test_command']}`.
- Synthetic gesture trials: generated landmarks check rule behavior, with IDLE and missing predictions preserved. They do not measure real lighting or distance robustness.
- Synthetic smoothing: replaying 360 generated points across three paths at alpha 1.0 and 0.5 reduced the jitter metric while increasing geometric error. It checks filter behavior, not user handwriting quality.
- Webcam field trials: {facts['camera_status']}.
- Live AI evaluation: {facts['live_ai_status']}.

The independent review reproduced six initial errors: failed API calls became prepared correct benchmark answers; incidental shared numbers became agreement; a rejected correct candidate passed the correctness grader; IDLE predictions earned HOVER credit; lost tracking preserved the clear timer; and provider failures rendered blank panels. The revised flow separates failures from results, compares explicit answers, keeps evaluation provenance, resets state on tracking loss, and displays provider errors.

{facts['repair_status']}

The earlier 100% accuracy claims and unsupported review-credit entries are not evidence for this report. Software tests do not establish real camera accuracy, live-model quality, response speed, or presentation frame rate.

## What is next

A clean installation and the camera setup still need validation on the presentation computer. Collect 25 observed trials per gesture with actual lighting and distance labels. Compare alpha 1.0 and 0.5 on the same captured fingertip sequence, including lag and corner shape. Run the same 15 actual board images through both live providers, retain full responses/model IDs/errors/timing, and grade them against a fixed human rubric. Count matching wrong answers separately from matching correct answers.

For the demonstration, show draw, hover, erase, clear, white background, and PNG saving. Use mouse mode if camera access fails. Show real AI answers only if requests succeed; otherwise explain the labeled demo and remaining work.

## Sources

- [MediaPipe Hands legacy solution](https://chuoling.github.io/mediapipe/solutions/hands.html)
- [OpenAI images and vision](https://developers.openai.com/api/docs/guides/images-vision?api-mode=chat)
- [Anthropic vision input guide](https://platform.claude.com/docs/en/build-with-claude/vision)
- [OpenCV image encoding](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html)

Configured draft repository: {LINK}. The repository contains the source code and submission materials. Put the actual repository link first in the online submission text. Due 2 November 2026 at 08:00, Asia/Taipei.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT/'docs/test_run.json')
    parser.add_argument('--output', type=Path, default=ROOT/'deliverables/airboard_midterm_report.pdf')
    parser.add_argument('--markdown', type=Path, default=ROOT/'deliverables/report.md')
    args = parser.parse_args()
    facts = content(read_json(args.evidence) if args.evidence else {})
    build_pdf(args.output, facts)
    args.markdown.write_text(markdown_report(facts))
    print(args.output)


if __name__ == '__main__':
    main()
