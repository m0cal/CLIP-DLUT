import base64
import html
import io
import os
import tempfile
import time
from typing import Any

import gradio as gr
import requests
from PIL import Image, ImageDraw

API_BASE = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 3600
DISPLAY_MAX_W = 1200
DISPLAY_MAX_H = 720
DISPLAY_BG = (0, 0, 0)

session = requests.Session()
session.trust_env = False


def _status_badge_html(ok: bool) -> str:
    dot_class = "status-dot online" if ok else "status-dot offline"
    text = "后端在线" if ok else "后端离线"
    return (
        "<div class='status-chip'>"
        f"<span class='{dot_class}'></span>"
        f"<span>{text}</span>"
        "</div>"
    )


def check_api_status() -> str:
    try:
        resp = session.get(f"{API_BASE}/", timeout=2.0)
        return _status_badge_html(resp.ok)
    except requests.RequestException:
        return _status_badge_html(False)


def _normalize_base64(data: str) -> str:
    if not data:
        return ""
    if "," in data:
        data = data.split(",", 1)[1]
    padding = (-len(data)) % 4
    if padding:
        data += "=" * padding
    return data


def _pil_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _base64_to_pil(data: str) -> Image.Image:
    raw = base64.b64decode(_normalize_base64(data))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _base64_to_cube_file(data: str) -> str:
    raw = base64.b64decode(_normalize_base64(data))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cube", prefix="clip_dlut_", mode="wb")
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return tmp.name


def _decode_image_payload(payload: Any) -> Image.Image | None:
    if payload is None:
        return None
    if isinstance(payload, Image.Image):
        return payload
    if isinstance(payload, str) and payload.strip():
        try:
            return _base64_to_pil(payload)
        except Exception:
            return None
    return None


def _fit_to_box(image: Image.Image, max_w: int = DISPLAY_MAX_W, max_h: int = DISPLAY_MAX_H) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    scale = min(max_w / width, max_h / height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    if new_w == width and new_h == height:
        return image
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def _pad_to_box(image: Image.Image, max_w: int = DISPLAY_MAX_W, max_h: int = DISPLAY_MAX_H) -> Image.Image:
    resized = _fit_to_box(image, max_w, max_h)
    canvas = Image.new("RGB", (max_w, max_h), DISPLAY_BG)
    offset_x = (max_w - resized.size[0]) // 2
    offset_y = (max_h - resized.size[1]) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _display_from_payload(payload: Any, fallback: Image.Image | None = None) -> Image.Image | None:
    img = _decode_image_payload(payload)
    if img is None:
        return fallback
    return _pad_to_box(img)


def _needs_final_image(original_b64: str | None, candidate_b64: str | None) -> bool:
    if not candidate_b64:
        return True
    original_img = _decode_image_payload(original_b64)
    candidate_img = _decode_image_payload(candidate_b64)
    if original_img is None or candidate_img is None:
        return True
    return (candidate_img.size[0] < original_img.size[0]) or (candidate_img.size[1] < original_img.size[1])


def _normalize_image_format(fmt: str) -> str:
    fmt_l = (fmt or "").strip().lower()
    if "jpg" in fmt_l or "jpeg" in fmt_l:
        return "jpg"
    return "png"


def _pil_to_download_file(image: Image.Image, fmt: str) -> str:
    ext = _normalize_image_format(fmt)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}", prefix="clip_dlut_", mode="wb")
    out = image.convert("RGB")
    if ext == "jpg":
        out.save(tmp, format="JPEG", quality=92, subsampling=1)
    else:
        out.save(tmp, format="PNG", compress_level=1)
    tmp.flush()
    tmp.close()
    return tmp.name


def _image_download_update(fmt: str, image_b64: str | None) -> Any:
    image_obj = _decode_image_payload(image_b64)
    if image_obj is None:
        return gr.update(value=None, visible=False, label="下载图片")
    path = _pil_to_download_file(image_obj, fmt)
    ext = _normalize_image_format(fmt)
    return gr.update(value=path, visible=True, label=f"下载图片 .{ext}")


def _friendly_error(message: str | None) -> str:
    if not message:
        return "处理失败，请稍后重试。"
    low = message.lower()
    if "timeout" in low or "timed out" in low:
        return "处理超时，请稍后重试。"
    if "cuda" in low or "out of memory" in low:
        return "显卡资源不足，请稍后重试。"
    if "ssl" in low or "connection" in low:
        return "网络连接不稳定，请稍后重试。"
    return "处理失败，请稍后重试。"


def _alert_html(message: str | None) -> str:
    if not message:
        return ""
    safe_text = html.escape(message)
    return f"<div class='alert-error'>{safe_text}</div>"


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "预计剩余：估算中"
    seconds_i = max(0, int(seconds))
    minute, sec = divmod(seconds_i, 60)
    if minute >= 60:
        hour, minute = divmod(minute, 60)
        return f"预计剩余：{hour}小时{minute}分"
    if minute > 0:
        return f"预计剩余：{minute}分{sec}秒"
    return f"预计剩余：{sec}秒"


def _estimate_eta(current: int, total: int, elapsed_sec: float) -> float | None:
    if current <= 5 or elapsed_sec <= 1:
        return None
    speed = current / elapsed_sec
    if speed <= 0:
        return None
    remain = (total - current) / speed
    return max(remain, 0.0)


def _poll_message(current: int, total: int, elapsed_sec: float, status: str = "processing") -> str:
    total = max(int(total), 1)
    current = max(0, min(int(current), total))
    pct = int((current / total) * 100)
    eta_text = _format_eta(_estimate_eta(current, total, elapsed_sec))
    if current >= total and status == "processing":
        eta_text = "正在生成最终图像"
    elif current == 0 and elapsed_sec >= 8:
        eta_text = "模型加载中，进度即将开始"

    return (
        "<div class='progress-wrap'>"
        "<div class='progress-title'>图像优化进度</div>"
        f"<div class='progress-value'>{current}/{total}</div>"
        "<div class='progress-track'>"
        f"<div class='progress-fill' style='width:{max(pct, 3)}%;'></div>"
        "</div>"
        f"<div class='progress-task'>{eta_text}</div>"
        "</div>"
    )


def _success_html() -> str:
    return (
        "<div class='success-card'>"
        "<div class='success-title'>优化完成</div>"
        "<div class='success-subtitle'>结果已就绪，可直接下载图片与 LUT 文件。</div>"
        "</div>"
    )


def _finalizing_html() -> str:
    return (
        "<div class='success-card'>"
        "<div class='success-title'>正在生成最终图像</div>"
        "<div class='success-subtitle'>结果即将就绪，请稍候。</div>"
        "</div>"
    )


def _view_badge_html(mode: str) -> str:
    text = "当前查看：原图" if mode == "original" else "当前查看：优化图"
    return f"<div class='view-badge'>{text}</div>"


def _compose_display_image(view_mode: str, original_b64: str | None, result_b64: str | None) -> Image.Image | None:
    if view_mode == "original" or not result_b64:
        return _decode_image_payload(original_b64)
    out = _decode_image_payload(result_b64)
    if out is not None:
        return out
    return _decode_image_payload(original_b64)


def _build_compare_image(original_b64: str | None, result_b64: str | None, split_value: float) -> Image.Image | None:
    original_img = _display_from_payload(original_b64)
    result_img = _display_from_payload(result_b64)
    if original_img is None or result_img is None:
        return None

    if original_img.size != result_img.size:
        original_img = original_img.resize(result_img.size, Image.Resampling.LANCZOS)
    width, height = result_img.size
    try:
        split_pct = float(split_value)
    except (TypeError, ValueError):
        split_pct = 50.0
    split = int(width * max(0.0, min(100.0, split_pct)) / 100.0)
    mixed = result_img.copy()
    if split > 0:
        left = original_img.crop((0, 0, split, height))
        mixed.paste(left, (0, 0))

    draw = ImageDraw.Draw(mixed)
    draw_x = min(max(split, 0), max(width - 1, 0))
    draw.line([(draw_x, 0), (draw_x, height)], fill=(230, 230, 230), width=2)
    return mixed


def sync_compare(
    original_b64: str | None,
    result_b64: str | None,
    split_value: float,
) -> Image.Image | None:
    return _build_compare_image(original_b64, result_b64, split_value)


def toggle_view_mode(current_mode: str) -> tuple[str, str, Any, Any]:
    new_mode = "original" if current_mode == "result" else "result"
    show_original = new_mode == "original"
    return (
        new_mode,
        _view_badge_html(new_mode),
        gr.update(visible=show_original),
        gr.update(visible=not show_original),
    )


def run_retouch_flow(
    image: Image.Image | None,
    target_prompt: str,
    original_prompt: str,
    iteration: int,
    image_format: str,
) -> Any:
    def _result_pil(original_image: Image.Image | None, result_b64: str | None) -> Image.Image | None:
        fallback = _pad_to_box(original_image) if original_image is not None else None
        return _display_from_payload(result_b64, fallback=fallback)

    def _original_pil(original_image: Image.Image | None, original_b64: str | None) -> Image.Image | None:
        fallback = _pad_to_box(original_image) if original_image is not None else None
        return _display_from_payload(original_b64, fallback=fallback)

    def _compare_pil(original_b64: str | None, result_b64: str | None) -> Image.Image | None:
        return _build_compare_image(original_b64, result_b64, 50)


    if image is None:
        yield (
            "<div class='progress-wrap'><div class='progress-title'>等待输入图片</div></div>",
            _alert_html("请先上传图片。"),
            gr.skip(),
            gr.skip(),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="开始优化", interactive=True),
            "",
            gr.update(value="切换图像", interactive=False),
            gr.skip(),
            gr.skip(),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
            None,
        )
        return

    if not target_prompt.strip():
        yield (
            "<div class='progress-wrap'><div class='progress-title'>等待输入目标描述</div></div>",
            _alert_html("请输入目标描述。"),
            gr.skip(),
            gr.skip(),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="开始优化", interactive=True),
            "",
            gr.update(value="切换图像", interactive=False),
            gr.skip(),
            gr.skip(),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
            None,
        )
        return

    input_b64 = _pil_to_base64_png(image)

    try:
        submit_resp = session.post(
            f"{API_BASE}/retouch",
            json={
                "image": input_b64,
                "target_prompt": target_prompt,
                "original_prompt": original_prompt,
                "iteration": int(iteration),
            },
            timeout=30,
        )
        submit_resp.raise_for_status()
        task_id = submit_resp.json()["task_id"]
    except requests.RequestException:
        yield (
            "<div class='progress-wrap'><div class='progress-title'>任务提交失败</div></div>",
            _alert_html("请求失败，请确认后端已启动。"),
            input_b64,
            None,
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="开始优化", interactive=True),
            "",
            gr.update(value="切换图像", interactive=False),
            gr.skip(),
            gr.skip(),
            gr.update(value=_original_pil(image, input_b64), visible=False),
            gr.update(value=_result_pil(image, input_b64), visible=True),
            _compare_pil(input_b64, input_b64),
        )
        return

    latest_b64 = input_b64

    yield (
        _poll_message(0, int(iteration), 0),
        "",
        input_b64,
        latest_b64,
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="处理中...", interactive=False),
        "",
        gr.update(value="切换图像", interactive=True),
        gr.skip(),
        gr.skip(),
        gr.update(value=_original_pil(image, input_b64), visible=False),
        gr.update(value=_result_pil(image, latest_b64), visible=True),
        _compare_pil(input_b64, latest_b64),
    )

    start_ts = time.time()
    while True:
        elapsed = time.time() - start_ts
        if elapsed > POLL_TIMEOUT_SECONDS:
            yield (
                _poll_message(0, int(iteration), elapsed),
                _alert_html("任务轮询超时，请稍后重试。"),
                input_b64,
                latest_b64,
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="开始优化", interactive=True),
                "",
                gr.update(value="切换图像", interactive=True),
                gr.skip(),
                gr.skip(),
                _original_pil(image, input_b64),
                _result_pil(image, latest_b64),
                _compare_pil(input_b64, latest_b64),
            )
            return

        try:
            query_resp = session.post(
                f"{API_BASE}/query_task",
                json={"task_id": task_id, "include_image": True},
                timeout=45,
            )
            query_resp.raise_for_status()
            data = query_resp.json()
        except requests.RequestException:
            yield (
                _poll_message(0, int(iteration), elapsed),
                _alert_html("网络暂时不稳定，请稍后重试。"),
                input_b64,
                latest_b64,
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="开始优化", interactive=True),
                "",
                gr.update(value="切换图像", interactive=True),
                gr.skip(),
                gr.skip(),
                _original_pil(image, input_b64),
                _result_pil(image, latest_b64),
                _compare_pil(input_b64, latest_b64),
            )
            return

        status = str(data.get("status", "")).lower()
        current = int(data.get("current_iteration") or 0)
        overall = int(data.get("overall_iteration") or iteration or 1000)

        maybe_b64 = data.get("image")
        if _decode_image_payload(maybe_b64) is not None:
            latest_b64 = maybe_b64

        if status in ("pending", "processing"):
            yield (
                _poll_message(current, overall, elapsed, status=status),
                "",
                gr.skip(),
                latest_b64,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
                gr.update(value="切换图像", interactive=True),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                _result_pil(image, latest_b64),
                _compare_pil(input_b64, latest_b64),
            )
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if status == "finished":
            # Ensure final image exists (avoid keeping a low-res preview).
            needs_final = _needs_final_image(input_b64, latest_b64)
            if needs_final:
                # Yield quickly with the best preview we have to avoid UI "freezing".
                yield (
                    _poll_message(overall, overall, elapsed, status="finished"),
                    "",
                    gr.skip(),
                    latest_b64,
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value="开始优化", interactive=True),
                    _finalizing_html(),
                    gr.update(value="切换图像", interactive=True),
                    "result",
                    _view_badge_html("result"),
                    gr.update(value=_original_pil(image, input_b64), visible=False),
                    gr.update(value=_result_pil(image, latest_b64), visible=True),
                    _compare_pil(input_b64, latest_b64),
                )

                try:
                    final_resp = session.post(
                        f"{API_BASE}/query_task",
                        json={"task_id": task_id, "include_image": True},
                        timeout=20,
                    )
                    final_resp.raise_for_status()
                    final_json = final_resp.json()
                    final_img = _decode_image_payload(final_json.get("image"))
                    if final_img is not None:
                        latest_b64 = final_json.get("image")
                    data.update(final_json)
                except requests.RequestException:
                    pass

            if _decode_image_payload(latest_b64) is None:
                yield (
                    _poll_message(overall, overall, elapsed, status="finished"),
                    _alert_html("任务已完成，但结果图暂时无法读取，请重试。"),
                    input_b64,
                    input_b64,
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value="开始优化", interactive=True),
                    "",
                    gr.update(value="切换图像", interactive=True),
                    "result",
                    _view_badge_html("result"),
                    gr.update(value=_original_pil(image, input_b64), visible=False),
                    gr.update(value=_result_pil(image, input_b64), visible=True),
                    _compare_pil(input_b64, input_b64),
                )
                return

            # Get LUT only after the final image is already painted.
            yield (
                _poll_message(overall, overall, elapsed, status="finished"),
                "",
                gr.skip(),
                latest_b64,
                _image_download_update(image_format, latest_b64),
                gr.update(value=None, visible=False),
                gr.update(value="开始优化", interactive=True),
                _success_html(),
                gr.update(value="切换图像", interactive=True),
                "result",
                _view_badge_html("result"),
                gr.update(value=_original_pil(image, input_b64), visible=False),
                gr.update(value=_result_pil(image, latest_b64), visible=True),
                _compare_pil(input_b64, latest_b64),
            )

            cube_file = None
            lut_b64 = data.get("lut")
            if not lut_b64:
                try:
                    lut_resp = session.post(
                        f"{API_BASE}/query_task",
                        json={"task_id": task_id, "include_image": False, "lut_format": "cube"},
                        timeout=8,
                    )
                    lut_resp.raise_for_status()
                    lut_json = lut_resp.json()
                    lut_b64 = lut_json.get("lut") if isinstance(lut_json, dict) else None
                except requests.RequestException:
                    lut_b64 = None

            if lut_b64:
                try:
                    cube_file = _base64_to_cube_file(lut_b64)
                except Exception:
                    cube_file = None

            yield (
                gr.skip(),
                "",
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.update(value=cube_file, visible=bool(cube_file)),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
            return

        if status == "failed":
            yield (
                _poll_message(current, overall, elapsed, status="failed"),
                _alert_html(_friendly_error(data.get("error"))),
                input_b64,
                latest_b64,
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="开始优化", interactive=True),
                "",
                gr.update(value="切换图像", interactive=True),
                gr.skip(),
                gr.skip(),
                _original_pil(image, input_b64),
                _result_pil(image, latest_b64),
                _compare_pil(input_b64, latest_b64),
            )
            return

        if status == "stopped":
            yield (
                _poll_message(current, overall, elapsed, status="stopped"),
                _alert_html("任务已停止。"),
                input_b64,
                latest_b64,
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="开始优化", interactive=True),
                "",
                gr.update(value="切换图像", interactive=True),
                gr.skip(),
                gr.skip(),
                _original_pil(image, input_b64),
                _result_pil(image, latest_b64),
                _compare_pil(input_b64, latest_b64),
            )
            return

        yield (
            _poll_message(current, overall, elapsed),
            _alert_html("任务状态未知，请稍后重试。"),
            input_b64,
            latest_b64,
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="开始优化", interactive=True),
            "",
            gr.update(value="切换图像", interactive=True),
            gr.skip(),
            gr.skip(),
            _original_pil(image, input_b64),
            _result_pil(image, latest_b64),
            _compare_pil(input_b64, latest_b64),
        )
        return


CUSTOM_CSS = """
:root {
  --bg: #000000;
  --panel: rgba(22, 22, 23, 0.82);
  --panel-strong: rgba(22, 22, 23, 0.94);
  --line: rgba(255, 255, 255, 0.10);
  --line-soft: rgba(255, 255, 255, 0.06);
  --text: #e6e6e6;
  --muted: #9aa1ad;
  --blue: #1e2a47;
  --blue-2: #3a4d76;
  --green: #32d74b;
  --red: #ff4d4f;
}

html, body, .gradio-container {
  background: radial-gradient(1200px 600px at 10% -20%, rgba(58,77,118,0.12), transparent 60%), var(--bg) !important;
  color: var(--text) !important;
  font-family: "Roboto", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif !important;
}

.gradio-container {
  width: 1440px !important;
  max-width: 1440px !important;
  margin: 0 auto !important;
  padding-top: 20px !important;
}

@media (max-width: 1500px) {
  .gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    padding: 12px !important;
  }
}

#top-bar, #left-card, #right-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 1.5rem;
  backdrop-filter: blur(20px);
}

#top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  margin-bottom: 16px;
}

.logo {
  font-size: 1.1rem;
  letter-spacing: 0.06em;
  font-weight: 650;
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 0.92rem;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 9999px;
  box-shadow: 0 0 8px currentColor;
}

.status-dot.online { color: var(--green); background: var(--green); }
.status-dot.offline { color: var(--red); background: var(--red); }

#left-card, #right-card {
  padding: 18px;
  min-height: 760px;
}

#left-card h3, #right-card h3 {
  margin: 0 0 8px 0;
  font-size: 0.96rem;
  color: var(--muted);
  font-weight: 600;
}

#upload-zone {
  border: 1px dashed rgba(255,255,255,0.22) !important;
  border-radius: 1rem !important;
  background: rgba(255,255,255,0.01) !important;
}

#left-card .gr-button.primary {
  background: linear-gradient(120deg, var(--blue) 0%, var(--blue-2) 100%) !important;
  border: none !important;
  color: #eff3ff !important;
  border-radius: 999px !important;
  min-height: 44px !important;
  font-weight: 620 !important;
  box-shadow: 0 10px 24px rgba(30, 42, 71, 0.45);
}

.progress-wrap {
  border: 1px solid var(--line-soft);
  border-radius: 1rem;
  background: var(--panel-strong);
  padding: 12px 14px;
}

.progress-title {
  color: #c8cbd1;
  font-size: 0.86rem;
  margin-bottom: 2px;
}

.progress-value {
  font-size: 1rem;
  font-weight: 620;
}

.progress-track {
  margin-top: 8px;
  width: 100%;
  height: 6px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
}

.progress-fill {
  position: relative;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--blue) 0%, var(--blue-2) 100%);
  box-shadow: 0 0 12px rgba(58, 77, 118, 0.5);
  transition: width .15s linear;
}

.progress-task {
  color: var(--muted);
  font-size: 0.8rem;
  margin-top: 4px;
}

.alert-error {
  border: 1px solid rgba(255, 77, 79, 0.35);
  background: rgba(255, 77, 79, 0.12);
  color: #ffc5c6;
  border-radius: 1rem;
  padding: 12px 14px;
  font-size: 0.92rem;
  white-space: pre-wrap;
}

#right-image-wrap {
  border: 1px solid var(--line-soft);
  border-radius: 1rem;
  padding: 10px;
  min-height: 680px;
  background: rgba(255,255,255,0.02);
}

#right-image-wrap img,
#right-image-wrap canvas {
  width: 100% !important;
  height: 100% !important;
  object-fit: contain !important;
}

#compare-wrap img,
#compare-wrap canvas {
  width: 100% !important;
  height: 100% !important;
  object-fit: contain !important;
}

.download-btn {
  border-radius: 999px !important;
  border: 1px solid var(--line) !important;
  background: rgba(30,42,71,0.35) !important;
  color: #d5deef !important;
}

#action-row {
  align-items: center !important;
  gap: 10px !important;
}

#format-select {
  min-width: 170px !important;
  max-width: 210px !important;
}

.view-badge {
  margin: 6px 0 10px 0;
  color: var(--muted);
  font-size: 0.88rem;
}

#compare-wrap {
  margin-top: 12px;
  border: 1px solid var(--line-soft);
  border-radius: 1rem;
  padding: 12px;
  background: rgba(255,255,255,0.02);
}

.success-card {
  margin-top: 12px;
  border: 1px solid rgba(80, 180, 120, 0.36);
  background: rgba(50, 160, 90, 0.14);
  border-radius: 1rem;
  padding: 12px 14px;
  animation: cardIn .35s ease-out;
}

.success-title {
  color: #d2f5df;
  font-size: 0.98rem;
  font-weight: 640;
}

.success-subtitle {
  color: #b6d8c2;
  font-size: 0.85rem;
  margin-top: 2px;
}

@keyframes cardIn {
  0% { opacity: 0; transform: translateY(6px); }
  100% { opacity: 1; transform: translateY(0); }
}
"""


with gr.Blocks(title="CLIP-DLUT 调色台", delete_cache=(3600, 3600)) as demo:
    with gr.Row(elem_id="top-bar"):
        gr.HTML("<div class='logo'>CLIP-DLUT</div>")
        api_badge = gr.HTML(value=check_api_status())
    heartbeat = gr.Timer(value=8.0, active=True)

    with gr.Row(equal_height=True):
        with gr.Column(scale=3, min_width=360, elem_id="left-card"):
            gr.HTML("<h3>操作面板</h3>")
            image_input = gr.Image(type="pil", label="上传图片", elem_id="upload-zone", height=240)
            target_prompt = gr.Textbox(
                label="目标色调",
                placeholder="输入目标色调描述，例如：赛博朋克蓝调、电影感高对比",
                lines=3,
            )
            original_prompt = gr.Textbox(
                label="原始描述",
                placeholder="输入原始图像描述，例如：一条巷子，白天自然光",
                value="一张自然色调的图片",
                lines=2,
            )
            iteration = gr.Slider(minimum=1, maximum=5000, value=1000, step=1, label="优化轮次")
            run_btn = gr.Button("开始优化", variant="primary")
            progress_html = gr.HTML("<div class='progress-wrap'><div class='progress-title'>等待任务</div></div>")
            error_html = gr.HTML("")

        with gr.Column(scale=7, min_width=720, elem_id="right-card"):
            gr.HTML("<h3>结果预览</h3>")
            with gr.Row(elem_id="action-row"):
                toggle_btn = gr.Button("切换图像", variant="secondary", interactive=True)
                image_format = gr.Dropdown(
                    choices=["PNG（无损）", "JPG（高兼容）", "JPEG（高兼容）"],
                    value="PNG（无损）",
                    show_label=False,
                    elem_id="format-select",
                )
                image_download = gr.DownloadButton(
                    label="下载图片",
                    value=None,
                    visible=False,
                    elem_classes=["download-btn"],
                )
                cube_download = gr.DownloadButton(
                    label="下载 LUT (.cube)",
                    value=None,
                    visible=False,
                    elem_classes=["download-btn"],
                )
            view_badge = gr.HTML(_view_badge_html("result"))

            with gr.Column(elem_id="right-image-wrap"):
                original_image = gr.Image(type="pil", label=None, height=720, visible=False)
                result_image = gr.Image(type="pil", label=None, height=720, visible=True)

            with gr.Column(elem_id="compare-wrap"):
                compare_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="滑块对比（左原图 / 右优化图）")
                compare_image = gr.Image(type="pil", label=None, height=420)
                completion_html = gr.HTML("")

    original_b64_state = gr.State(value=None)
    result_b64_state = gr.State(value=None)
    view_mode_state = gr.State(value="result")

    run_btn.click(
        fn=run_retouch_flow,
        inputs=[image_input, target_prompt, original_prompt, iteration, image_format],
        outputs=[
            progress_html,
            error_html,
            original_b64_state,
            result_b64_state,
            image_download,
            cube_download,
            run_btn,
            completion_html,
            toggle_btn,
            view_mode_state,
            view_badge,
            original_image,
            result_image,
            compare_image,
        ],
        show_progress="hidden",
    )

    toggle_evt = toggle_btn.click(
        fn=toggle_view_mode,
        inputs=[view_mode_state],
        outputs=[view_mode_state, view_badge, original_image, result_image],
        queue=False,
        show_progress="hidden",
    )

    compare_slider.change(
        fn=sync_compare,
        inputs=[original_b64_state, result_b64_state, compare_slider],
        outputs=[compare_image],
        queue=False,
        show_progress="hidden",
    )
    result_b64_state.change(
        fn=sync_compare,
        inputs=[original_b64_state, result_b64_state, compare_slider],
        outputs=[compare_image],
        queue=False,
        show_progress="hidden",
    )

    image_format.change(
        fn=_image_download_update,
        inputs=[image_format, result_b64_state],
        outputs=[image_download],
        queue=False,
        show_progress="hidden",
    )

    heartbeat.tick(fn=check_api_status, outputs=[api_badge], queue=False)

demo.queue(default_concurrency_limit=2, max_size=16)

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, css=CUSTOM_CSS, max_threads=8)
