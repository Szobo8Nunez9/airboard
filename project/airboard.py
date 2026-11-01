"""Webcam air drawing with optional vision assistance and a mouse demo mode."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import os
import sys
import time

import cv2
import numpy as np

from core.hand_tracking import HandTracker
from core.gestures import GestureClassifier, GestureMode
from core.smoothing import ExponentialSmoother
from core.canvas import AirCanvas
from agents import AgentResult, ask_both, display_response
from core.notebook import Notebook
from core.handwriting import recognize_handwriting
from core.notebook_export import export_notebook_pdf
from core import notebook_view


class AirboardApp:
    def __init__(self, camera_id=0, width=1280, height=720, alpha=0.5, mock_ai=False, mouse=False, adaptive=False, notebook=False, notebook_dir=None):
        if width < 800 or height < 600:
            raise ValueError('Use a window at least 800 x 600 pixels')
        self.width, self.height = width, height
        self.alpha = alpha
        self.mock_ai = mock_ai
        if mock_ai:
            os.environ['MOCK_VISION_APIS'] = 'true'
        self.mouse_control = mouse
        self.pointer = None
        self.pointer_draw = False
        self.pointer_erase = False
        self.tracker = None
        self.classifier = GestureClassifier(clear_hold_seconds=1.0)
        self.smoother = ExponentialSmoother(alpha, adaptive=adaptive)
        self.canvas = AirCanvas(width, height)
        self.canvas.whiteboard_mode = mouse or notebook
        self.notebook_folder = Path(notebook_dir) if notebook_dir else Path(__file__).parent / 'notebooks' / 'default'
        self.book = Notebook.load(self.notebook_folder, width, height) if notebook else Notebook(width, height)
        self.canvas.canvas = self.book.current.ink
        self.notebook_view = notebook
        self.notebook_used = notebook
        self.notebook_editing = False
        self.notebook_scroll = 0
        self.notebook_notice = ''
        self.notebook_future = None
        self.notebook_job = None
        self.last_notebook_save = time.monotonic()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.ai_future = None
        self.ai_loading = False
        self.ai_results = None
        self.ai_panel_visible = False
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.snapshots_dir = Path(__file__).parent / 'snapshots'

    def _open_camera(self, index):
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == 'darwin' else [cv2.CAP_ANY]
        for backend in backends:
            candidate = cv2.VideoCapture(index, backend)
            if candidate.isOpened():
                candidate.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return candidate
            candidate.release()
        return None

    def _ensure_tracker(self):
        if self.tracker is None:
            self.tracker = HandTracker(min_detection_confidence=0.7, min_tracking_confidence=0.6)

    def start_camera(self):
        if self.mouse_control:
            return False
        self.cap = self._open_camera(self.camera_id)
        if self.cap is None:
            self.mouse_control = True
            self.canvas.whiteboard_mode = True
            print('[airboard] Camera unavailable. Mouse demo mode: left drag draws, right drag erases.')
            return False
        try:
            self._ensure_tracker()
        except RuntimeError:
            self.cap.release()
            self.cap = None
            self.mouse_control = True
            self.canvas.whiteboard_mode = True
            print('[airboard] Hand tracking could not start. Mouse demo mode is active.')
            return False
        return True

    def switch_camera(self):
        # A failed candidate never closes the currently working camera.
        offsets = range(1, 5) if self.mouse_control else range(1, 4)
        for offset in offsets:
            index = (self.camera_id + offset) % 4
            candidate = self._open_camera(index)
            if candidate is None:
                continue
            try:
                self._ensure_tracker()
            except RuntimeError:
                candidate.release()
                print('[airboard] Hand tracking is unavailable. Keeping the current input mode.')
                return False
            if self.cap is not None:
                self.cap.release()
            self.cap = candidate
            self.camera_id = index
            self.mouse_control = False
            self.canvas.reset_stroke()
            self.smoother.reset()
            self.classifier.classify([], time.monotonic())
            self.pointer_draw = self.pointer_erase = False
            return True
        print('[airboard] No alternate camera opened. Keeping the current input mode.')
        return False

    def get_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return cv2.resize(cv2.flip(frame, 1), (self.width, self.height))
            # A dropped frame must count as lost hand tracking, not old landmarks.
        return np.full((self.height, self.width, 3), 235, np.uint8)

    def on_mouse(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN and self.ai_panel_visible:
            self.ai_panel_visible = False
            self.pointer_draw = False
            self.canvas.reset_stroke()
            return
        if self.notebook_view:
            _, _, ink, buttons = notebook_view.layout(self.width, self.height, self.book.width, self.book.height)
            if event == cv2.EVENT_LBUTTONDOWN:
                for label, key, (bx, by, bw, bh) in buttons:
                    if bx <= x <= bx+bw and by <= y <= by+bh:
                        self.pointer_draw = self.pointer_erase = False
                        self.canvas.reset_stroke()
                        if self.notebook_editing:
                            self.finish_notebook_edit()
                        self.handle_key(ord(key))
                        return
            if event == cv2.EVENT_LBUTTONUP:
                self.pointer_draw = False
                self.canvas.reset_stroke()
            elif event == cv2.EVENT_RBUTTONUP:
                self.pointer_erase = False
            if not self.mouse_control:
                return
            ix, iy, iw, ih = ink
            if not (ix <= x < ix+iw and iy <= y < iy+ih):
                self.pointer = None
                return
            x = round((x-ix) * self.width / iw)
            y = round((y-iy) * self.height / ih)
        if not self.mouse_control:
            return
        self.pointer = (max(0, min(self.width - 1, x)), max(0 if self.notebook_view else 90, min(self.height - 1, y)))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pointer_draw = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.pointer_draw = False
            self.canvas.reset_stroke()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pointer_erase = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.pointer_erase = False

    def execute_mode(self, mode, point):
        if mode in (GestureMode.DRAW, GestureMode.ERASE, GestureMode.CLEAR):
            self.book.mark_ink_changed()
        if mode == GestureMode.CLEAR:
            self.canvas.clear()
            self.smoother.reset()
            return point
        if mode == GestureMode.DRAW and point is not None:
            if self.canvas.prev_point is None:
                self.smoother.reset()
            tip = self.smoother.smooth(*point)
            self.canvas.draw_segment(self.canvas.prev_point or tip, tip)
            self.canvas.prev_point = tip
            return tip
        self.canvas.reset_stroke()
        self.smoother.reset()
        if mode == GestureMode.ERASE and point is not None:
            self.canvas.erase_at(point)
        return point

    def process_landmarks(self, landmarks, now):
        # Always advance the state machine, including when tracking disappears.
        if self.ai_panel_visible or self.notebook_editing:
            landmarks = []
        mode, progress, _ = self.classifier.classify(landmarks, current_time=now)
        point = None
        if len(landmarks) >= 21:
            if mode == GestureMode.ERASE:
                point = ((landmarks[8][0]+landmarks[12][0])//2, (landmarks[8][1]+landmarks[12][1])//2)
            elif mode == GestureMode.HOVER:
                point = landmarks[9]
            else:
                point = landmarks[8]
        return mode, progress, self.execute_mode(mode, point)

    def process_pointer(self, now):
        self.classifier.classify([], current_time=now)
        if self.ai_panel_visible or self.notebook_editing or self.pointer is None:
            mode = GestureMode.IDLE
        elif self.pointer_erase:
            mode = GestureMode.ERASE
        elif self.pointer_draw:
            mode = GestureMode.DRAW
        else:
            mode = GestureMode.HOVER
        return mode, 0.0, self.execute_mode(mode, self.pointer)

    def trigger_ai_query(self):
        if self.ai_loading or self.canvas.is_empty():
            return
        png = self.canvas.get_png_bytes()
        self.ai_loading = True
        self.ai_panel_visible = True
        self.ai_results = None
        self.canvas.reset_stroke()
        self.pointer_draw = False
        self.ai_future = self.executor.submit(ask_both, png)

    def poll_ai_results(self):
        if self.ai_loading and self.ai_future is not None and self.ai_future.done():
            try:
                self.ai_results = self.ai_future.result()
            except Exception as error:
                result = AgentResult('Vision', '', 0, False, str(error), mode='error')
                self.ai_results = result, result, None
            self.ai_loading = False
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            path = self.snapshots_dir / 'last_ai_answers.txt'
            try:
                path.write_text('\n\n'.join(
                    f'{r.agent_name} ({r.model}, {r.mode})\n{display_response(r)}'
                    for r in self.ai_results[:2]
                ), encoding='utf-8')
            except OSError:
                print('[airboard] Could not save the full answer text.')

    @staticmethod
    def _text(img, text, point, scale=0.6, color=(35, 35, 35), thickness=1):
        cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def render_hud(self, display, mode, progress, fingertip):
        if self.notebook_view:
            source = ('MOUSE' if self.mouse_control else f'WEBCAM {self.camera_id}') + ' ' + mode.value
            display[:] = notebook_view.render(self.book, display, source, self.notebook_notice,
                                             self.notebook_editing, self.notebook_scroll, fingertip, progress, mode == GestureMode.ERASE)
            return
        display[:86, :] = (244, 244, 244)
        self._text(display, 'airboard', (22, 33), 0.85, (120, 75, 15), 2)
        source = 'MOUSE DEMO' if self.mouse_control else f'WEBCAM {self.camera_id}'
        self._text(display, f'{source}   {mode.value}   alpha {self.smoother.alpha:.2f}', (200, 33), 0.56)
        self._text(display, 'WHITE BOARD' if self.canvas.whiteboard_mode else 'VIDEO OVERLAY', (self.width - 210, 33), 0.5)
        self._text(display, 'Tab Camera   [ ] Smoothing   a AI   w Board   c Clear   s Save   1-5 Colors   b Notebook   x Close   q Quit', (22, 65), 0.46 if self.width < 1000 else 0.52)
        if progress > 0:
            x, y, width = self.width // 2 - 160, self.height - 60, 320
            cv2.rectangle(display, (x, y), (x + width, y + 15), (180, 180, 180), -1)
            cv2.rectangle(display, (x, y), (x + int(width * progress), y + 15), (60, 60, 220), -1)
            self._text(display, 'Hold fist continuously to clear', (x, y - 12), 0.5)
        if fingertip is not None and not self.ai_panel_visible:
            radius = self.canvas.eraser_radius if mode == GestureMode.ERASE else 8
            cv2.circle(display, fingertip, radius, self.canvas.current_color, 2, cv2.LINE_AA)

    def render_ai_panel(self, display):
        if not self.ai_panel_visible:
            return
        x, y = 35, 105
        width, height = self.width - 70, self.height - 140
        cv2.rectangle(display, (x, y), (x + width, y + height), (248, 248, 248), -1)
        self._text(display, 'Board snapshot   x or click to close', (x + 18, y + 30), 0.6)
        if self.ai_loading:
            self._text(display, 'Waiting for vision providers...', (x + 25, y + 100), 0.75)
            return
        if self.ai_results is None:
            return
        claude, openai, agreement = self.ai_results
        if agreement is None:
            status = 'Comparison unavailable - read each result'
        elif agreement:
            status = 'Matching final answers - correctness still needs checking'
        else:
            status = 'Different final answers - compare the reasoning'
        self._text(display, status, (x + 18, y + 62), 0.5, (120, 75, 15))
        col_width = (width - 60) // 2
        for index, result in enumerate((claude, openai)):
            cx = x + 20 + index * (col_width + 20)
            self._text(display, f'{result.agent_name}   {result.latency_seconds:.2f}s', (cx, y + 105), 0.68, thickness=2)
            self._text(display, f'{result.mode.upper()}   {result.model}', (cx, y + 132), 0.46)
            max_lines = max(3, (height - 165) // 22)
            self._wrap_text(display, display_response(result), cx, y + 165, col_width, max_lines)

    def _wrap_text(self, img, text, x, y, max_width_px, max_lines=15):
        # OpenCV's built-in font is ASCII. Keep math symbols readable where possible.
        text = text.replace('−', '-').replace('×', '*').replace('÷', '/').replace('²', '^2')
        text = text.encode('ascii', 'replace').decode('ascii')
        lines = []
        for paragraph in text.splitlines():
            line = ''
            for word in paragraph.split():
                trial = (line + ' ' + word).strip()
                if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] > max_width_px and line:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
        if len(lines) > max_lines:
            lines = lines[:max_lines - 1] + ['[Full reply: snapshots/last_ai_answers.txt]']
        for index, line in enumerate(lines):
            self._text(img, line, (x, y + index * 22), 0.5)

    def save_notebook(self):
        if not self.notebook_used:
            return
        try:
            self.book.save(self.notebook_folder)
            self.last_notebook_save = time.monotonic()
        except OSError:
            self.notebook_notice = 'Could not save the notebook. Your pages remain open; try another writable folder.'

    def bind_notebook_page(self):
        self.canvas.canvas = self.book.current.ink
        self.canvas.reset_stroke()
        self.smoother.reset()
        self.classifier.classify([], time.monotonic())
        self.pointer_draw = self.pointer_erase = False
        self.notebook_scroll = 0
        self.ai_panel_visible = False

    def trigger_tidy_text(self):
        self.notebook_view = self.notebook_used = True
        if self.notebook_future is not None:
            self.notebook_notice = 'Still reading a page. You can keep writing or turn pages.'
            return
        if self.canvas.is_empty():
            self.notebook_notice = 'Write some words first, or choose Edit text to type.'
            return
        self.notebook_job = (self.book.current.id, self.book.current.ink_hash())
        self.notebook_future = self.executor.submit(recognize_handwriting, self.canvas.get_png_bytes())
        self.canvas.reset_stroke()
        self.notebook_notice = 'Reading handwriting locally. First use may take a moment.'

    def poll_notebook(self):
        if self.notebook_future is None or not self.notebook_future.done():
            return
        try:
            result = self.notebook_future.result()
            page_id, digest = self.notebook_job
            applied = self.book.apply_transcription(page_id, digest, result)
            page_number = next((i+1 for i,p in enumerate(self.book.pages) if p.id == page_id), '?')
            self.notebook_notice = (f'Page {page_number} text is ready. Compare it with your handwriting; Edit text corrects words.'
                                    if applied else result.get('notice', 'No readable words found. Your existing text is kept.'))
            if applied and self.book.find(page_id).stale:
                self.notebook_notice = f'Page {page_number} changed while reading. Tidy again to include the newer ink.'
            self.save_notebook()
        except Exception as error:
            self.notebook_notice = str(error)[:140]
        finally:
            self.notebook_future = self.notebook_job = None

    def finish_notebook_edit(self):
        self.notebook_editing = False
        self.notebook_notice = 'Edited wording saved. Original ink is kept.'
        self.save_notebook()

    def export_notebook(self):
        self.notebook_used = True
        self.save_notebook()
        try:
            path = self.notebook_folder.parent / 'exports' / f'notebook_{datetime.now():%Y%m%d_%H%M%S}.pdf'
            export_notebook_pdf(self.book, path)
            self.notebook_notice = f'PDF saved: notebooks/exports/{path.name}'
            print(f'[airboard] Notebook PDF saved: {path}')
        except Exception:
            self.notebook_notice = 'PDF export failed. Your notebook and original ink are kept.'

    def handle_key(self, key):
        if self.notebook_editing:
            page = self.book.current
            if key == 27:
                self.finish_notebook_edit()
            elif key in (8, 127):
                page.text = page.text[:-1]
            elif key in (10, 13):
                page.text += '\n'
            elif 32 <= key <= 126:
                page.text += chr(key)
            page.edited = True
            return True
        if key == ord('b'):
            self.notebook_view = not self.notebook_view
            self.notebook_used = True
            self.canvas.reset_stroke()
            self.smoother.reset()
            self.pointer_draw = self.pointer_erase = False
            self.save_notebook()
            return True
        if key == ord('t'):
            self.trigger_tidy_text()
            return True
        if key == ord('n'):
            self.notebook_view = self.notebook_used = True
            self.book.new_page()
            self.bind_notebook_page()
            self.notebook_notice = 'New page. Previous pages are kept.'
            self.save_notebook()
            return True
        if key in (ord(','), ord('.')):
            self.notebook_view = self.notebook_used = True
            self.book.turn(-1 if key == ord(',') else 1)
            self.bind_notebook_page()
            self.save_notebook()
            return True
        if key == ord('e'):
            self.notebook_view = self.notebook_used = True
            if self.notebook_future is not None:
                self.notebook_notice = 'Wait for local recognition to finish before editing text.'
                return True
            page = self.book.current
            page.text_history = (page.text_history + [page.text])[-10:]
            self.notebook_editing = True
            self.pointer_draw = self.pointer_erase = False
            self.canvas.reset_stroke()
            return True
        if key == ord('u') and self.notebook_view:
            page = self.book.current
            if page.text_history:
                page.text = page.text_history.pop()
                page.edited = True
                self.notebook_notice = 'Previous text restored.'
                self.save_notebook()
            return True
        if key in (ord('j'), ord('k')) and self.notebook_view:
            self.notebook_scroll = max(0, self.notebook_scroll + (3 if key == ord('j') else -3))
            return True
        if key == ord('f'):
            self.export_notebook()
            return True
        if key in (27, ord('q')):
            return False
        if key == 9:
            self.switch_camera()
        elif key in (ord('['), ord(']')):
            delta = -0.05 if key == ord('[') else 0.05
            self.smoother.set_alpha(max(0.1, min(1.0, round(self.smoother.alpha + delta, 2))))
        elif key == ord('a'):
            self.trigger_ai_query()
        elif key == ord('w'):
            self.canvas.whiteboard_mode = not self.canvas.whiteboard_mode
        elif key == ord('c'):
            self.book.mark_ink_changed()
            self.canvas.clear()
            self.smoother.reset()
            self.classifier.classify([], time.monotonic())
        elif key == ord('s'):
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            name = self.snapshots_dir / f'airboard_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
            self.canvas.save_snapshot(str(name))
            print(f'[airboard] Saved {name}')
        elif key in (ord('x'), 32):
            self.ai_panel_visible = False
            self.pointer_draw = False
        elif ord('1') <= key <= ord('5'):
            self.canvas.set_color(list(self.canvas.COLOR_PALETTE)[key - ord('1')])
        return True

    def close(self):
        self.save_notebook()
        if self.cap is not None:
            self.cap.release()
        if self.tracker is not None:
            self.tracker.close()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def run(self):
        window = 'airboard'
        try:
            self.start_camera()
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, self.on_mouse)
            self.running = True
            while self.running:
                self.poll_notebook()
                if self.notebook_used and time.monotonic() - self.last_notebook_save >= 15:
                    self.save_notebook()
                frame = self.get_frame()
                now = time.monotonic()
                if self.mouse_control:
                    mode, progress, tip = self.process_pointer(now)
                else:
                    self.tracker.find_hands(frame, draw=False)
                    landmarks, _, _ = self.tracker.get_landmark_positions(frame)
                    mode, progress, tip = self.process_landmarks(landmarks, now)
                display = self.canvas.render_view(frame)
                self.render_hud(display, mode, progress, tip)
                self.poll_ai_results()
                self.render_ai_panel(display)
                cv2.imshow(window, display)
                self.running = self.handle_key(cv2.waitKey(1) & 0xFF)
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description='Draw in the air with a webcam, or use the mouse demo.')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--notebook', action='store_true', help='Open saved notebook pages with local handwriting recognition.')
    parser.add_argument('--notebook-dir', help='Folder for local notebook pages (default: notebooks/default).')
    parser.add_argument('--adaptive', action='store_true', help='Optional speed-dependent smoothing; omit for fixed EMA comparisons.')
    parser.add_argument('--mock-ai', action='store_true', help='Show labeled demo replies. No API calls or image interpretation.')
    parser.add_argument('--mouse', action='store_true', help='Mouse drawing demo: left drag draws, right drag erases.')
    args = parser.parse_args()
    try:
        AirboardApp(args.camera, args.width, args.height, args.alpha, args.mock_ai, args.mouse, args.adaptive, args.notebook, args.notebook_dir).run()
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f'airboard: {error}\n')


if __name__ == '__main__':
    main()
