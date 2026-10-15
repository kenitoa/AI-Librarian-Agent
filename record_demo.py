"""Record only the actual demo HWND with Windows Graphics Capture.

Optional capture environment: windows-capture==2.0.1, imageio-ffmpeg, Pillow.
The agent environment and requirements stay independent. The target may be
occluded but must remain unminimized so Windows keeps rendering. No desktop
fallback is permitted. Actual captured frames are sampled at real-time CFR.
"""
import argparse
import ctypes
from ctypes import wintypes
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def main():
    from PIL import Image, ImageDraw
    from windows_capture import WindowsCapture

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="AI Librarian Agent - CMD Demo")
    parser.add_argument("--signal", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-seconds", type=float, help="Actual-window probe only, no model execution")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if not args.probe_seconds and (not args.signal or not args.result):
        parser.error("--signal and --result are required for a full live demo")
    if args.output.exists() or (args.signal and args.signal.exists()) or (args.result and args.result.exists()):
        parser.error("Use fresh paths to preserve previous evidence")
    ffmpeg = os.environ.get("FFMPEG_EXE") or shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.signal:
        args.signal.parent.mkdir(parents=True, exist_ok=True)
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowW.restype = wintypes.HWND
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    target = user32.FindWindowW(None, args.title)
    if not target:
        raise RuntimeError("Exact demo HWND not found; desktop fallback is forbidden")
    if user32.IsIconic(target):
        raise RuntimeError("Demo is minimized; leave it unminimized (it may be behind other apps)")

    def title_matches():
        title = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(target, title, len(title))
        return title.value == args.title

    lock = threading.Lock()
    first, closed = threading.Event(), threading.Event()
    current = {"pixels": None, "size": None, "count": 0, "failure": None}
    capture = WindowsCapture(window_hwnd=int(target), cursor_capture=False,
                             secondary_window=False, draw_border=True, minimum_update_interval=50)

    @capture.event
    def on_frame_arrived(frame, control):
        if not title_matches():
            current["failure"] = "Demo title changed; refusing to record a different tab"
            control.stop()
            return
        size = (frame.width, frame.height)
        with lock:
            if current["size"] and size != current["size"]:
                current["failure"] = "Demo dimensions changed during capture"
                control.stop()
                return
            current.update(pixels=frame.frame_buffer.tobytes(order="C"), size=size, count=current["count"] + 1)
        first.set()

    @capture.event
    def on_closed():
        closed.set()

    control = capture.start_free_threaded()
    if not first.wait(15):
        control.stop()
        raise TimeoutError("No frame supplied by the specific demo HWND")
    width, height = current["size"]
    initial = Image.frombytes("RGBA", (width, height), current["pixels"], "raw", "BGRA").convert("RGB")
    if initial.getextrema()[0][1] - initial.getextrema()[0][0] < 12:
        control.stop()
        raise RuntimeError("Selected window supplied a blank frame")
    initial.save(args.output.with_suffix(".first.png"))
    report, case_events = {}, []
    frame_count, background_samples = 0, 0
    failure, finished_seen = None, None
    signalled, last_poll = False, 0.0
    started = time.monotonic()
    log_path = args.output.with_suffix(".capture.log")
    with log_path.open("w", encoding="utf-8") as log:
        encoder = subprocess.Popen([
            ffmpeg, "-hide_banner", "-f", "rawvideo", "-pixel_format", "bgra",
            "-video_size", f"{width}x{height}", "-framerate", str(args.fps), "-i", "pipe:0",
            "-an", "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-threads", "4",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ], stdin=subprocess.PIPE, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while True:
                elapsed = time.monotonic() - started
                if current["failure"] or closed.is_set() or user32.IsIconic(target) or not title_matches():
                    raise RuntimeError(current["failure"] or "Demo HWND closed, minimized, or changed title")
                if encoder.poll() is not None:
                    raise RuntimeError("FFmpeg stopped; inspect capture log")
                with lock:
                    pixels = current["pixels"]
                encoder.stdin.write(pixels)
                frame_count += 1
                if elapsed - last_poll >= 0.2:
                    last_poll = elapsed
                    background_samples += int(user32.GetForegroundWindow() != target)
                    if args.signal and not signalled and elapsed >= 1:
                        args.signal.write_text("specific HWND recording started\n", encoding="utf-8")
                        signalled = True
                        print("WGC specific-window recording started; live runner released", flush=True)
                    if args.result and args.result.exists():
                        try:
                            report = json.loads(args.result.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            pass
                        if len(report.get("cases", [])) > len(case_events):
                            for item in report["cases"][len(case_events):]:
                                case_events.append({"id": item["case"]["id"], "passed": item["passed"],
                                                    "report_seen_seconds": round(elapsed, 3),
                                                    "video_sample_seconds": round(frame_count / args.fps + 2, 3)})
                            print(f"Actual live cases completed: {len(case_events)}/22", flush=True)
                    if report.get("finished_at") and finished_seen is None:
                        finished_seen = elapsed
                if args.probe_seconds and elapsed >= args.probe_seconds:
                    break
                if finished_seen is not None and elapsed >= finished_seen + 7:
                    break
                if elapsed > 900:
                    raise TimeoutError("Live runner did not finish within 15 minutes")
                time.sleep(max(0, started + frame_count / args.fps - time.monotonic()))
        except BaseException as exc:
            failure = exc
        finally:
            control.stop()
            control.wait()
            with lock:
                final_pixels = current["pixels"]
            Image.frombytes("RGBA", (width, height), final_pixels, "raw", "BGRA").convert("RGB").save(args.output.with_suffix(".last.png"))
            encoder.stdin.close()
            encoder.wait(timeout=40)
    summary = {"video": str(args.output.resolve()), "source": "actual specific CMD HWND via Windows Graphics Capture",
               "capture_library": "windows-capture==" + importlib.metadata.version("windows-capture"),
               "hwnd": int(target), "title": args.title, "source_size": [width, height], "output_size": [1920, 1080],
               "fps": args.fps, "encoded_frames": frame_count, "video_seconds": round(frame_count / args.fps, 3),
               "wall_seconds": round(time.monotonic() - started, 3), "source_frames_arrived": current["count"],
               "background_capture_samples": background_samples, "foreground_required": False,
               "secondary_windows": False, "desktop_captured": False, "probe_only": bool(args.probe_seconds),
               "result": str(args.result.resolve()) if args.result else None, "bytes": args.output.stat().st_size,
               "passed": report.get("passed"), "total": report.get("total"), "case_events": case_events,
               "failure": str(failure) if failure else None}
    metadata_path = args.output.with_suffix(".recording.json")
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failure:
        raise failure
    if encoder.returncode:
        raise RuntimeError("FFmpeg encoding failed; inspect " + str(log_path))
    decoded = subprocess.run([ffmpeg, "-v", "error", "-i", str(args.output), "-f", "null", "NUL"],
                             capture_output=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
    summary["full_decode_passed"] = decoded.returncode == 0 and not decoded.stderr
    if not summary["full_decode_passed"]:
        raise RuntimeError("Full decode failed: " + decoded.stderr.decode("utf-8", "replace"))
    if case_events:
        sheet_dir = args.output.parent / (args.output.stem + "-frames")
        sheet_dir.mkdir(exist_ok=True)
        sheet = Image.new("RGB", (1920, ((len(case_events) + 3) // 4) * 292), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(case_events):
            frame_path = sheet_dir / f"{index + 1:02d}-{item['id']}.png"
            subprocess.run([ffmpeg, "-v", "error", "-ss", str(item["video_sample_seconds"]), "-i", str(args.output),
                            "-frames:v", "1", str(frame_path)], check=True, capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            with Image.open(frame_path) as frame:
                thumbnail = frame.resize((480, 270), Image.Resampling.LANCZOS)
                x, y = (index % 4) * 480, (index // 4) * 292
                sheet.paste(thumbnail, (x, y))
                draw.text((x + 4, y + 272), f"{index + 1:02d} {item['id']} {item['video_sample_seconds']:.1f}s", fill="black")
            item["decoded_video_frame"] = str(frame_path.resolve())
        contact_path = args.output.with_suffix(".contact.png")
        sheet.save(contact_path)
        summary["contact_sheet"] = str(contact_path.resolve())
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if args.probe_seconds or report.get("passed") == report.get("total") == 22 else 1


if __name__ == "__main__":
    raise SystemExit(main())
