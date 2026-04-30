import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Callable

import boto3


class VideoIntelError(Exception):
    pass


def slug_from_title(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:80]


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise VideoIntelError(f"'{name}' not found in PATH")
    return found


def upload_to_r2(local_dir: Path, r2_prefix: str):
    CONTENT_TYPES = {
        ".json": "application/json",
        ".md": "text/markdown",
        ".jpg": "image/jpeg",
    }
    SKIP = {"source.mp4", "audio.mp3"}

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET"]

    for path in local_dir.rglob("*"):
        if path.is_file() and path.name not in SKIP:
            rel = path.relative_to(local_dir).as_posix()
            key = f"{r2_prefix}/{rel}"
            ct = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
            with open(path, "rb") as f:
                s3.put_object(Bucket=bucket, Key=key, Body=f, ContentType=ct)
            print(f"  uploaded: {key}", flush=True)


def process_video(url: str, user_id: str, progress: Callable[[str, int], None]) -> str:
    from faster_whisper import WhisperModel

    yt_dlp = find_tool("yt-dlp")
    ffmpeg = find_tool("ffmpeg")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = tmp / "out"
        out.mkdir()

        # Bypass YouTube bot-detection from datacenter IPs
        ytdlp_extractor_args = ["--extractor-args", "youtube:player_client=tv,web_safari,ios,android"]

        # Stage 1 — fetch metadata
        progress("downloading", 5)
        meta_r = subprocess.run(
            [yt_dlp, "--dump-json", "--no-download", *ytdlp_extractor_args, url],
            capture_output=True, text=True,
        )
        if meta_r.returncode != 0:
            raise VideoIntelError(f"Metadata failed: {meta_r.stderr.strip()}")
        meta = json.loads(meta_r.stdout)
        title = meta.get("title", "untitled")
        duration = int(meta.get("duration", 0) or 0)

        # Stage 1 — download
        progress("downloading", 15)
        video_path = tmp / "source.mp4"
        dl_r = subprocess.run([
            yt_dlp,
            "--ffmpeg-location", str(Path(ffmpeg).parent),
            "-f", "best[height<=720][ext=mp4]/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "--merge-output-format", "mp4",
            "--no-playlist", "--socket-timeout", "30", "--retries", "5",
            *ytdlp_extractor_args,
            "-o", str(video_path), url,
        ], capture_output=True, text=True)
        if dl_r.returncode != 0:
            raise VideoIntelError(f"Download failed: {dl_r.stderr[-500:]}")

        progress("extracting", 35)

        # Stage 2 — audio + frames in parallel
        def extract_audio():
            ap = out / "audio.mp3"
            subprocess.run([
                ffmpeg, "-y", "-i", str(video_path),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(ap),
            ], capture_output=True, check=True)
            return ap

        def extract_frames():
            fd = out / "frames"
            fd.mkdir()
            subprocess.run([
                ffmpeg, "-y", "-i", str(video_path),
                "-vf", "fps=1/10",
                str(fd / "frame_%04d.jpg"),
            ], capture_output=True, check=True)
            frames = sorted(fd.glob("*.jpg"))
            index = [{"file": f.name, "start": i * 10, "end": (i + 1) * 10} for i, f in enumerate(frames)]
            (fd / "index.json").write_text(json.dumps(index, indent=2))

        with ThreadPoolExecutor(max_workers=2) as pool:
            af = pool.submit(extract_audio)
            pool.submit(extract_frames)
            audio_path = af.result()

        progress("transcribing", 50)

        # Stage 3 — transcribe
        model = WhisperModel("base", compute_type="int8", num_workers=2, cpu_threads=2)
        segments_iter, info = model.transcribe(
            str(audio_path),
            beam_size=1, best_of=1, temperature=0,
            vad_filter=True, condition_on_previous_text=False,
        )

        lines = ["# Transcript\n"]
        segs = []
        for i, seg in enumerate(segments_iter):
            s = str(timedelta(seconds=int(seg.start)))
            e = str(timedelta(seconds=int(seg.end)))
            lines.append(f"**[{s} → {e}]** {seg.text.strip()}\n")
            segs.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            if i % 10 == 0 and duration:
                pct = 50 + int((seg.end / duration) * 30)
                progress("transcribing", min(pct, 80))

        (out / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
        (out / "segments.json").write_text(json.dumps(segs, indent=2), encoding="utf-8")

        progress("packaging", 82)

        # Stage 4 — package
        slug = slug_from_title(title)
        sensitive = {"http_headers", "cookies", "cookie", "downloader_options"}
        clean_meta = {k: v for k, v in meta.items() if k not in sensitive}
        (out / "metadata.json").write_text(json.dumps(clean_meta, indent=2, default=str), encoding="utf-8")

        ud = meta.get("upload_date", "")
        created = f"{ud[:4]}-{ud[4:6]}-{ud[6:8]}" if len(ud) == 8 else ""
        summary = "\n".join([
            f"# {title}\n",
            f"**Duration:** {timedelta(seconds=duration)}",
            f"**Source:** {meta.get('webpage_url', url)}",
            f"**Uploader:** {meta.get('uploader', 'Unknown')}",
            f"**Date:** {created}",
        ])
        (out / "summary.md").write_text(summary, encoding="utf-8")

        progress("uploading", 85)

        # Stage 5 — upload to R2
        upload_to_r2(out, slug)

        progress("done", 100)
        return slug
