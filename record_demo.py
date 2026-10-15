"""Record an already-open CMD demonstration window; never synthesize model output.

Optional recording dependency: imageio-ffmpeg, or set FFMPEG_EXE to official FFmpeg.
The live runner must wait for --signal and write --result when finished.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="AI Librarian Agent - CMD Demo")
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.signal.exists() or args.result.exists() or args.output.exists():
        parser.error("Choose unused signal, result, and output paths to preserve earlier evidence.")
    ffmpeg = os.environ.get("FFMPEG_EXE") or shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.signal.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".capture.log")
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    user32.FindWindowW.restype = wintypes.HWND
    user32.GetForegroundWindow.restype = wintypes.HWND
    target = user32.FindWindowW(None, args.title)
    if not target or not user32.IsWindowVisible(target) or user32.GetForegroundWindow() != target:
        raise RuntimeError("The requested CMD demo must be visible in the foreground before recording.")
    bounds = wintypes.RECT()
    if not user32.GetWindowRect(target, ctypes.byref(bounds)):
        raise RuntimeError("Cannot read the demo window bounds.")
    # Modern Windows Terminal renders a black GDI window surface. Capture only its
    # visible foreground region, inset to exclude transparent borders/background.
    crop = [bounds.left + 8, bounds.top + 8, bounds.right - bounds.left - 16, bounds.bottom - bounds.top - 16]
    with log_path.open("w", encoding="utf-8") as log:
        capture = subprocess.Popen([
            ffmpeg, "-hide_banner", "-f", "gdigrab", "-framerate", "15",
            "-draw_mouse", "0", "-offset_x", str(crop[0]), "-offset_y", str(crop[1]),
            "-video_size", f"{crop[2]}x{crop[3]}", "-i", "desktop",
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ], stdin=subprocess.PIPE, stdout=log, stderr=log)
        started = time.monotonic()
        failure = None
        try:
            time.sleep(2)
            if capture.poll() is not None:
                raise RuntimeError("Window capture did not start. Inspect " + str(log_path))
            args.signal.write_text("recording started\n", encoding="utf-8")
            print("Recording actual CMD window and live agent checks...", flush=True)
            report = {}
            while not report.get("finished_at"):
                if user32.GetForegroundWindow() != target or not user32.IsWindowVisible(target):
                    raise RuntimeError("Recording interrupted: the CMD demo lost the foreground. Do not publish this capture.")
                if capture.poll() is not None:
                    raise RuntimeError("Capture stopped before live results were saved.")
                if time.monotonic() - started > 900:
                    raise TimeoutError("Live runner did not finish within 15 minutes.")
                if args.result.exists():
                    try:
                        report = json.loads(args.result.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        pass  # The runner may currently be writing its progress file.
                time.sleep(1)
            time.sleep(7)  # Keep the actual final test summary readable in the video.
        except BaseException as exc:
            failure = exc
        finally:
            if capture.poll() is None:
                capture.communicate(input=b"q\n", timeout=30)
        if failure:
            raise failure
        if capture.returncode:
            raise RuntimeError("FFmpeg failed; inspect " + str(log_path))
    summary = {"video": str(args.output.resolve()), "seconds": round(time.monotonic() - started, 2),
               "source": "actual foreground CMD window region via FFmpeg gdigrab", "crop": crop,
               "result": str(args.result.resolve()),
               "bytes": args.output.stat().st_size,
               "passed": report.get("passed"), "total": report.get("total")}
    args.output.with_suffix(".recording.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if report.get("passed") == report.get("total") else 1


if __name__ == "__main__":
    raise SystemExit(main())
