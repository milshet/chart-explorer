#!/usr/bin/env python3
"""The demo video (task 4.8): generated, narrated segments around a live screen recording of the demo.

  python3 scripts/serve/video.py --segments                     # build/video/{intro,results,close}.mp4
  python3 scripts/serve/video.py --join ~/Desktop/live.mov      # build/video/demo_video.mp4: intro, live, results, close

Needs ffmpeg (`brew install ffmpeg`) and macOS `say`; nothing else, so it runs with the Mac's own python3. Slides are
drawn by ffmpeg itself.

The generated segments state only what the committed results say: their text comes from the same functions as
docs/demo_script.md (demo.headline(), demo.do_not_claim()), and the training-set size from data/v2.0/stats.json. The
live segment is the presenter's own screen recording of `demo.py --live` against the deployed endpoint, captioned
on screen as a live recording so a viewer can always tell recorded responses from generated slides.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import date
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OUT = REPO / "build" / "video"
W, H, FPS = 1920, 1080, 30
BG, FG, ACCENT, DIM = "0x0f172a", "0xf8fafc", "0x38bdf8", "0x94a3b8"
FONT = "/System/Library/Fonts/Helvetica.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
VOICE = "Samantha"


def tool(name):
    """ffmpeg/ffprobe with text drawing: Homebrew's plain ffmpeg has no drawtext, its keg-only ffmpeg-full does."""
    import os
    for d in (os.environ.get("FFMPEG_DIR"), "/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"):
        if d and (Path(d) / name).is_file():
            return str(Path(d) / name)
    return shutil.which(name) or name


# --- what the segments say ----------------------------------------------------------------------------------------

def spoken(text):
    """Text as the voice should read it: units, ratios and scientific notation spelled out."""
    t = re.sub(r"`([^`]*)`", r"\1", text)
    t = re.sub(r"\s*\((?:task\s*)?\d+\.\d+[a-z]?\)", "", t)     # "(3.0i)", "(task 4.4)": for slides, not for the voice
    t = t.replace("κ", "kappa").replace(" vs ", " versus ")
    t = re.sub(r"(\d(?:\.\d+)?)e-0*(\d+)", r"\1 times ten to the minus \2", t)
    t = re.sub(r"\bp = ", "p equals ", t)
    t = re.sub(r"(\d+)/(\d+)", r"\1 of \2", t)
    t = re.sub(r"(\d) ms\b", r"\1 milliseconds", t)
    t = re.sub(r"(\d) s\b", r"\1 seconds", t)
    t = re.sub(r"(\d)-point\b", r"\1 point", t)
    t = t.replace("Qwen2.5-VL-3B", "Qwen 2.5 V L, 3 B").replace("bf16", "B F 16").replace("vLLM", "V L L M")
    t = t.replace("L40S", "L 40 S").replace("Eval-C", "Eval C").replace("Eval-B", "Eval B").replace("—", ",")
    return t


def short(item):
    """A 'do not claim' item cut to its claim, for a slide bullet."""
    return re.split(r"[:;]", item, maxsplit=1)[0].strip()


def segments(root=REPO):
    """{segment: [slide, ...]}; a slide is {title, lines, narration}. Every number comes from committed files."""
    from serve import demo
    st = json.loads((Path(root) / "data/v2.0/stats.json").read_text())["splits"]["train"]
    head = demo.headline(root)
    claims = demo.do_not_claim(root)
    intro = [{"title": "Auditable Chart QA",
              "lines": ["Qwen2.5-VL-3B, fine-tuned with QLoRA on "
                        f"{st['kept']:,} blind-verified finance and economics chart questions",
                        "Answers with the evidence it read and its reasoning, and refuses when the chart cannot answer",
                        "Built with Qwen  ·  research and evaluation use only"],
              "narration": f"This is our capstone: auditable chart question answering. We fine-tuned Qwen 2.5 V L, a "
                           f"three billion parameter vision-language model, on {st['kept']:,} finance and economics "
                           "chart questions labelled blind. It answers with the evidence it read and its reasoning, "
                           "and it refuses when the chart cannot answer. What follows is the deployed endpoint, "
                           "recorded live."}]
    results = [{"title": "Measured results", "lines": head,
                "narration": "Here is what we measured, all on held-out data. " + " ".join(spoken(h) for h in head)}]
    half = (len(claims) + 1) // 2
    close = [{"title": "What we do not claim (1 of 2)", "lines": [short(c) for c in claims[:half]],
              "narration": "And here is what we do not claim. " + " ".join(f"Not {spoken(short(c))}." for c in claims[:half])},
             {"title": "What we do not claim (2 of 2)", "lines": [short(c) for c in claims[half:]],
              "narration": " ".join(f"Not {spoken(short(c))}." for c in claims[half:])
                           + " Every number in this video is read from the committed results. Thank you."}]
    return {"intro": intro, "results": results, "close": close}


# --- rendering ---------------------------------------------------------------------------------------------------

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(map(str, cmd[:6]))} ...\n{r.stderr[-1500:]}")
    return r.stdout


def has_audio(path):
    out = run([tool("ffprobe"), "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)])
    return bool(out.strip())


def slide_filter(slide, tmp, font=FONT):
    """A drawtext chain for one slide; each line's text goes through a file, so nothing needs escaping."""
    parts, y = [], 150
    def text(s, size, colour, y, fontfile=font):
        f = Path(tmp) / f"t{len(parts)}.txt"
        f.write_text(s, encoding="utf-8")
        parts.append(f"drawtext=fontfile='{fontfile}':textfile='{f}':expansion=none:fontsize={size}:fontcolor={colour}:x=140:y={y}")
    text(slide["title"], 76, ACCENT, y)
    y += 150
    for line in slide["lines"]:
        wrapped = textwrap.wrap(line, 70) or [""]
        for i, w in enumerate(wrapped):
            text(("•  " if i == 0 else "    ") + w, 40, FG, y)
            y += 54
        y += 24
    text("ChartAudit-Fin  ·  generated from the committed results", 28, DIM, H - 90)
    return ",".join(parts)


def render_slide(slide, path, tmp):
    audio = Path(tmp) / (path.stem + ".aiff")
    run(["say", "-v", VOICE, "-o", str(audio), slide["narration"]])
    run([tool("ffmpeg"), "-y", "-f", "lavfi", "-i", f"color=c={BG}:s={W}x{H}:r={FPS}", "-i", str(audio),
         "-filter_complex", f"[0:v]{slide_filter(slide, tmp)}[v];[1:a]apad=pad_dur=1.2,aresample=48000[a]",
         "-map", "[v]", "-map", "[a]", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-ar", "48000", "-ac", "2", str(path)])


def concat(parts, path, tmp):
    listing = Path(tmp) / f"{path.stem}.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
    run([tool("ffmpeg"), "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(path)])


def build_segments(out=OUT, root=REPO):
    out.mkdir(parents=True, exist_ok=True)
    built = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, slides in segments(root).items():
            parts = []
            for i, slide in enumerate(slides):
                p = Path(tmp) / f"{name}_{i}.mp4"
                render_slide(slide, p, tmp)
                parts.append(p)
            concat(parts, out / f"{name}.mp4", tmp)
            built[name] = out / f"{name}.mp4"
            print(f"wrote {built[name]}")
    return built


def normalise_live(src, dst, recorded_on):
    """The screen recording at the segments' size, rate and audio format, captioned as a live recording."""
    label = f"LIVE recording of the deployed endpoint  ·  {recorded_on}"
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "label.txt"
        f.write_text(label, encoding="utf-8")
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={BG},"
              f"fps={FPS},format=yuv420p,drawtext=fontfile='{FONT}':textfile='{f}':expansion=none:fontsize=30:fontcolor=white:"
              "box=1:boxcolor=0xdc2626@0.85:boxborderw=12:x=30:y=30")
        cmd = [tool("ffmpeg"), "-y", "-i", str(src)]
        if not has_audio(src):
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest", "-map", "0:v", "-map", "1:a"]
        run(cmd + ["-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
                   "-c:a", "aac", "-ar", "48000", "-ac", "2", str(dst)])


def join(live, out=OUT, recorded_on=None):
    seg = {n: out / f"{n}.mp4" for n in ("intro", "results", "close")}
    if not all(p.is_file() for p in seg.values()):
        build_segments(out)
    with tempfile.TemporaryDirectory() as tmp:
        live_n = Path(tmp) / "live.mp4"
        normalise_live(live, live_n, recorded_on or date.today().isoformat())
        final = out / "demo_video.mp4"
        concat([seg["intro"], live_n, seg["results"], seg["close"]], final, tmp)
    print(f"wrote {final}")
    return final


def duration(path):
    return float(run([tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]))


def place(cues, durations, gap=0.3):
    """Start times for spoken cues: each at its mark, or just after the previous one ends if they would overlap."""
    starts, end = [], 0.0
    for cue, d in zip(cues, durations):
        t = max(cue["at"], end + gap if starts else cue["at"])
        starts.append(round(t, 2))
        end = t + d
    return starts, end


def voiceover(video, cues_file, out):
    """Narrate a screen recording: speak each cue with `say` at its time, mix them, and lay them over the video
    (the last frame is held if the narration runs past the end)."""
    spec = json.loads(Path(cues_file).read_text(encoding="utf-8"))
    cues, voice = spec["cues"], spec.get("voice", VOICE)
    length = duration(video)
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i, cue in enumerate(cues):
            a = Path(tmp) / f"cue{i:02d}.aiff"
            run(["say", "-v", voice, "-o", str(a), cue["text"]])
            clips.append(a)
        durations = [duration(c) for c in clips]
        starts, end = place(cues, durations)
        for cue, t, d in zip(cues, starts, durations):
            moved = "" if abs(t - cue["at"]) < 0.01 else f"  (moved from {cue['at']} s: the previous line ran long)"
            print(f"  {t:6.1f}-{t + d:6.1f} s  {cue['step']:16} {cue['text'][:60]}{moved}")
        hold = max(0.0, end + 1.0 - length)
        inputs = [x for c in clips for x in ("-i", str(c))]
        delays = ";".join(f"[{i + 1}:a]adelay={int(t * 1000)}:all=1[a{i}]" for i, t in enumerate(starts))
        mix = "".join(f"[a{i}]" for i in range(len(clips)))
        graph = (f"{delays};{mix}amix=inputs={len(clips)}:normalize=0:duration=longest,aresample=48000[aout];"
                 f"[0:v]tpad=stop_mode=clone:stop_duration={hold:.2f},format=yuv420p[vout]")
        run([tool("ffmpeg"), "-y", "-i", str(video), *inputs, "-filter_complex", graph, "-map", "[vout]",
             "-map", "[aout]", "-c:v", "libx264", "-crf", "18", "-r", str(FPS), "-c:a", "aac", "-ar", "48000",
             "-ac", "2", "-shortest" if hold == 0 else "-y", str(out)])
    print(f"wrote {out} ({duration(out):.1f} s; narration ends at {end:.1f} s, video was {length:.1f} s)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", action="store_true", help="build the intro, results and close segments")
    ap.add_argument("--join", type=Path, help="a screen recording of demo.py --live to put between the segments")
    ap.add_argument("--recorded-on", help="the date shown on the live caption (default: today)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--print", action="store_true", help="print every slide's text and narration, render nothing")
    ap.add_argument("--voiceover", type=Path, help="a screen recording to narrate with --cues")
    ap.add_argument("--cues", type=Path, default=REPO / "docs/demo_voiceover.json", help="timed narration for --voiceover")
    args = ap.parse_args(argv)
    if args.print:
        for name, slides in segments().items():
            for s in slides:
                print(f"== {name}: {s['title']}\n" + "\n".join(f"  • {l}" for l in s["lines"]) + f"\n  [voice] {s['narration']}\n")
        return 0
    if not shutil.which("say"):
        sys.exit("macOS `say` not found: the narration needs a Mac")
    filters = subprocess.run([tool("ffmpeg"), "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    if " drawtext " not in filters:
        sys.exit("this ffmpeg cannot draw text: run `brew install ffmpeg-full` (it installs beside plain ffmpeg)")
    if args.voiceover:
        src = args.voiceover.expanduser()
        if not src.is_file():
            sys.exit(f"{src} does not exist")
        args.out.mkdir(parents=True, exist_ok=True)
        voiceover(src, args.cues, args.out / f"{src.stem}_voiced.mp4")
    if args.segments:
        build_segments(args.out)
    if args.join:
        if not args.join.expanduser().is_file():
            sys.exit(f"{args.join} does not exist")
        join(args.join.expanduser(), args.out, args.recorded_on)
    if not (args.segments or args.join or args.voiceover):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
