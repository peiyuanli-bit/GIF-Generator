import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time
import math
import base64
from PIL import Image, ImageFilter, ImageDraw, ImageFont, ImageChops
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

# --- 兼容性修复 ---
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# --- 配置项 ---
MAX_FILE_SIZE_MB = 1.02
MIN_DIMENSION_PX = 320
MAX_DIMENSION_PX = 1080
DEFAULT_BTN1_NAME = "btn_appstore.png"
DEFAULT_BTN2_NAME = "btn_googleplay.png"
ICON_FILENAME = "Logo.webp"

# ==========================================
# [配置] 多语言字典 (无Emoji，关键按钮加粗)
# ==========================================
TRANSLATIONS = {
    "zh": {
        "internal_notice": "仅限内部 GM / Agency 使用。",
        "sidebar_title": "素材配置",
        "lang_select": "Language / 语言",
        "up_app": "App Store 图标",
        "up_play": "Google Play 图标",
        "up_sticker": "其他贴纸 (可选)",
        "main_title": "GIF 生成器",
        "step1_title": "1. 导入视频",
        "up_video": "上传 MP4 视频",
        "using_cached_video": "✅ 使用上次上传的视频",
        "step2_title": "2. 参数设置",
        "duration": "时长 (秒)",
        "scale": "主体大小占比",
        "btn_find": "寻找合适片段",
        "analyzing": "**正在分析视频内容...**",
        "error_video": "视频时长不足。",
        "step3_title": "选择片段",
        "previewing": "生成预览中...",
        "clip_label": "片段",
        "confirm_label": "确认使用片段：",
        # [修改] 去除 Emoji，使用 Markdown 加粗
        "btn_reselect": "**换一批片段**",
        "btn_generate": "**开始生成 GIF**",
        "balancing": "正在平衡画质与体积...",
        "success": "生成完成 | 体积: {size}MB | 尺寸: {w}x{h}",
        "warning_best": "已生成最佳结果，体积略大于预期。",
        "download": "下载 GIF",
        # [修改] 去除 Emoji，使用 Markdown 加粗
        "btn_restart": "**重新开始 (保留视频)**",
        "btn_clear_reset": "**完全重置 (清除视频)**",
        "err_file": "文件未生成",
        "err_0kb": "生成文件异常 (0KB)",
        "err_prev": "预览加载失败: {e}",
        "err_fail": "处理失败: {e}"
    },
    "en": {
        "internal_notice": "Internal Use Only (GM / Agency).",
        "sidebar_title": "Assets Config",
        "lang_select": "Language / 语言",
        "up_app": "App Store Icon",
        "up_play": "Google Play Icon",
        "up_sticker": "Extra Sticker (Optional)",
        "main_title": "GIF Generator",
        "step1_title": "1. Import Video",
        "up_video": "Upload MP4 Video",
        "using_cached_video": "✅ Using previously uploaded video",
        "step2_title": "2. Settings",
        "duration": "Duration (sec)",
        "scale": "Subject Scale",
        "btn_find": "Find Highlights",
        "analyzing": "**Analyzing video content...**",
        "error_video": "Video duration insufficient.",
        "step3_title": "Select Clip",
        "previewing": "Generating previews...",
        "clip_label": "Clip",
        "confirm_label": "Confirm Selection:",
        "btn_reselect": "**Reselect Clips**",
        "btn_generate": "**Generate GIF**",
        "balancing": "Balancing quality and size...",
        "success": "Done | Size: {size}MB | Dim: {w}x{h}",
        "warning_best": "Generated best result (slightly over target size).",
        "download": "Download GIF",
        "btn_restart": "**Restart (Keep Video)**",
        "btn_clear_reset": "**Full Reset (Clear Video)**",
        "err_file": "File not generated",
        "err_0kb": "Error: 0KB File",
        "err_prev": "Preview failed: {e}",
        "err_fail": "Process failed: {e}"
    }
}

st.set_page_config(page_title="GIF Generator", layout="wide")

# ==========================================
# [侧边栏] Header & Config
# ==========================================
sidebar_header_container = st.sidebar.container()
st.sidebar.markdown("### Language")
language_option = st.sidebar.radio("Select Language / 选择语言", ["中文", "English"], label_visibility="collapsed")
lang_code = "zh" if language_option == "中文" else "en"
t = TRANSLATIONS[lang_code]

with sidebar_header_container:
    if os.path.exists(ICON_FILENAME):
        st.image(ICON_FILENAME, width=260)
    else:
        st.header("MOLOCO")
    st.markdown(f"""
        <div style="margin-top: 5px; margin-bottom: 20px; text-align: center;">
            <span style="color: #333; font-weight: bold; font-size: 16px;">{t['internal_notice']}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

# ==========================================
# [State] 状态初始化
# ==========================================
if 'step' not in st.session_state: st.session_state.step = 1
if 'candidates' not in st.session_state: st.session_state.candidates = []
if 'temp_video_path' not in st.session_state: st.session_state.temp_video_path = None
# 新增：用于记录已经排除（出现过）的片段
if 'excluded_clips' not in st.session_state: st.session_state.excluded_clips = []
if 'upload_key' not in st.session_state: st.session_state.upload_key = str(time.time())


# ==========================================
# [功能函数]
# ==========================================

def save_upload_to_temp(uploaded_file, prefix):
    if uploaded_file is None: return None
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        if not data: return None
        ext = os.path.splitext(uploaded_file.name)[1] or ".png"
        temp_path = os.path.join(tempfile.gettempdir(), f"safe_{prefix}_{int(time.time())}{ext}")
        with open(temp_path, "wb") as f:
            f.write(data)
        return temp_path
    except Exception:
        return None


def get_image_path(upload_obj, default_filename, placeholder_text, target_height):
    if upload_obj is not None:
        path = save_upload_to_temp(upload_obj, placeholder_text)
        if path and os.path.exists(path): return path
    if os.path.exists(default_filename): return default_filename
    return create_adaptive_placeholder(placeholder_text, target_height)


def trim_transparent_borders(image_path):
    try:
        pil_img = Image.open(image_path)
        img = pil_img.convert("RGBA")
        bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox: return img.crop(bbox)
        return img
    except:
        return Image.new('RGB', (100, 100), (50, 50, 50))


def create_adaptive_placeholder(text, target_height):
    width = int(target_height * 3.2)
    height = target_height
    img = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, width, height), radius=int(height * 0.2), fill=(50, 50, 50))
    try:
        font = ImageFont.truetype("arial.ttf", int(height * 0.4))
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) / 2, (height - th) / 2), text, font=font, fill=(200, 200, 200))
    path = os.path.join(tempfile.gettempdir(), f"dummy_{text}_{height}.png")
    img.save(path)
    return path


def smart_export_gif(clip, output_path, max_mb=1.0):
    strategies = [
        (1.0, 20, 256), (1.0, 15, 192), (1.0, 15, 128),
        (0.95, 15, 128), (0.9, 12, 128), (0.85, 12, 96),
        (0.8, 10, 64), (0.7, 8, 32)
    ]
    original_w = clip.w
    original_h = clip.h
    min_edge = min(original_w, original_h)
    absolute_min_scale = MIN_DIMENSION_PX / min_edge if min_edge > 0 else 1.0

    msg = st.empty()
    progress_bar = st.progress(0)
    st.caption(t["balancing"])

    for i, (scale, fps, colors) in enumerate(strategies):
        effective_scale = max(scale, absolute_min_scale)
        effective_scale = min(effective_scale, 1.0)
        target_w = int(original_w * effective_scale)
        target_h = int(original_h * effective_scale)
        progress = (i + 1) / len(strategies)
        progress_bar.progress(progress)

        current_clip = clip.resize(width=target_w) if effective_scale != 1.0 else clip
        try:
            current_clip.write_gif(output_path, fps=fps, colors=colors, verbose=False, logger=None)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                if size_mb <= max_mb:
                    progress_bar.empty()
                    msg.empty()
                    st.success(t["success"].format(size=f"{size_mb:.2f}", w=target_w, h=target_h))
                    return True
        except:
            continue

    progress_bar.empty()
    msg.warning(t["warning_best"])
    return True


# --- 核心修改：支持排除片段的搜索算法 ---
def find_top_3_highlights(video_path, clip_duration=3.0, exclude_timestamps=[]):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step = 10
    search_len = total - int(clip_duration * fps)

    scores, timestamps = [], []
    prev = None

    status = st.empty()
    status.markdown(t["analyzing"])
    prog = st.progress(0)

    # 1. 计算所有帧的分数
    for i in range(0, search_len, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        score = 0
        if prev is not None: score = np.sum(cv2.absdiff(prev, gray))
        scores.append(score)
        timestamps.append(i / fps)
        prev = gray
        if i % (sample_step * 20) == 0: prog.progress(min(i / search_len, 1.0))

    cap.release()
    prog.empty()
    status.empty()

    scores = np.array(scores)
    timestamps = np.array(timestamps)

    # 2. 先把需要排除的时间段分数置为 -1
    exclusion_radius = clip_duration * 0.8  # 避免重复的半径

    for ex_t in exclude_timestamps:
        mask_exclude = np.abs(timestamps - ex_t) < exclusion_radius
        scores[mask_exclude] = -1

    top_clips = []

    # 3. 寻找 Top 3
    for _ in range(3):
        if len(scores) == 0 or np.max(scores) <= 0: break
        best_idx = np.argmax(scores)
        best_time = timestamps[best_idx]
        top_clips.append(best_time)
        # 找到一个后，把这个附近的也排除掉
        mask = np.abs(timestamps - best_time) < exclusion_radius
        scores[mask] = -1

    return sorted(top_clips)


def find_colorful_timestamp(video_path, sample_step=30):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_score, best_idx = -1, total - 1
    for i in range(0, total, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret: break
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        score = np.mean(hsv[:, :, 1]) + np.mean(hsv[:, :, 2])
        if score > max_score: max_score, best_idx = score, i
    cap.release()
    return best_idx / fps


def add_pulse_effect(clip, speed=2.0, amplitude=0.05):
    return clip.resize(lambda t: 1 + amplitude * math.sin(speed * np.pi * t))


def create_blurred_background(frame_img, size):
    pil_img = Image.fromarray(frame_img)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=15))
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 50))
    blurred.paste(overlay, (0, 0), overlay)
    bg_w, bg_h = pil_img.size
    target_w, target_h = size
    scale = max(target_w / bg_w, target_h / bg_h)
    blurred = blurred.resize((int(bg_w * scale), int(bg_h * scale)), Image.LANCZOS)
    left, top = (blurred.width - target_w) / 2, (blurred.height - target_h) / 2
    return ImageClip(np.array(blurred.crop((left, top, left + target_w, top + target_h))))


def show_gif_robust(file_path):
    if not os.path.exists(file_path):
        st.error(t["err_file"])
        return
    if os.path.getsize(file_path) < 100:
        st.error(t["err_0kb"])
        return
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64_data = base64.b64encode(data).decode()
        html_code = f'''
        <div style="display: flex; justify-content: center; margin-top: 20px; margin-bottom: 20px;">
            <img src="data:image/gif;base64,{b64_data}" 
                 style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        </div>
        '''
        st.markdown(html_code, unsafe_allow_html=True)
    except Exception as e:
        st.error(t["err_prev"].format(e=e))


# ==========================================
# [Main UI]
# ==========================================

# 侧边栏：素材上传
st.sidebar.markdown(f"### {t['sidebar_title']}")
allowed_types = ["png", "jpg", "webp"]
btn1_up = st.sidebar.file_uploader(t["up_app"], type=allowed_types)
btn2_up = st.sidebar.file_uploader(t["up_play"], type=allowed_types)
extra_img_up = st.sidebar.file_uploader(t["up_sticker"], type=allowed_types)

st.title(t["main_title"])

# --- 步骤 1: 导入与设置 ---
if st.session_state.step == 1:
    st.markdown(f"#### {t['step1_title']}")

    # 1. 显示已有缓存状态
    if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
        st.info(f"{t['using_cached_video']}")

    # 2. 允许上传新文件
    uploaded_file = st.file_uploader(t["up_video"], type=["mp4"], key=st.session_state.upload_key,
                                     label_visibility="collapsed")

    st.markdown(f"#### {t['step2_title']}")
    col1, col2 = st.columns(2)
    with col1:
        gif_duration = st.number_input(t["duration"], value=2.5, step=0.5, max_value=5.0)
    with col2:
        scale_factor = st.slider(t["scale"], 0.6, 1.0, 0.9)

    st.markdown("---")

    # 逻辑：只要有上传文件 OR 有缓存文件，就可以开始
    can_proceed = uploaded_file is not None or (st.session_state.temp_video_path is not None)

    if can_proceed:
        if st.button(t["btn_find"], type="primary", use_container_width=True):
            # 优先处理新上传的文件
            if uploaded_file:
                video_path = save_upload_to_temp(uploaded_file, "source_video")
                if video_path:
                    st.session_state.temp_video_path = video_path
                    # 上传新视频时，清空之前排除列表
                    st.session_state.excluded_clips = []

            # 使用现有路径
            current_path = st.session_state.temp_video_path

            if current_path and os.path.exists(current_path):
                # 初始搜索（无排除）
                candidates = find_top_3_highlights(current_path, gif_duration, exclude_timestamps=[])
                if not candidates:
                    st.error(t["error_video"])
                else:
                    st.session_state.candidates = candidates
                    st.session_state.gif_duration = gif_duration
                    st.session_state.scale_factor = scale_factor
                    st.session_state.step = 2
                    st.rerun()

# --- 步骤 2: 预览与导出 ---
elif st.session_state.step == 2:
    st.markdown(f"#### {t['step3_title']}")
    video_path = st.session_state.temp_video_path
    duration = st.session_state.gif_duration
    candidates = st.session_state.candidates

    cols = st.columns(3)
    with st.spinner(t["previewing"]):
        original_clip = VideoFileClip(video_path)
        for i, timestamp in enumerate(candidates):
            with cols[i]:
                p_path = os.path.join(tempfile.gettempdir(), f"preview_{i}_{int(time.time())}.mp4")
                if not os.path.exists(p_path):
                    sub = original_clip.subclip(timestamp, timestamp + duration)
                    sub = sub.resize(height=240)
                    sub.write_videofile(p_path, codec="libx264", audio=False, preset="ultrafast", logger=None)
                st.video(p_path)
                st.caption(f"{t['clip_label']} {i + 1} ({timestamp:.1f}s)")
        original_clip.close()

    st.markdown("---")

    # 确认选择
    st.markdown(
        """<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>""",
        unsafe_allow_html=True
    )
    selected_option = st.radio(t["confirm_label"], [0, 1, 2],
                               format_func=lambda x: f"{t['clip_label']} {x + 1}",
                               label_visibility="collapsed")

    st.markdown("")

    # [修改] 布局优化：大方、直接、不扭捏
    # 将 "换一批" 和 "生成" 放在同一行，比例 1:2，让生成按钮更突出
    col_act1, col_act2 = st.columns([1, 2])

    with col_act1:
        # 换一批：Secondary 样式，填满宽度
        if st.button(t["btn_reselect"], type="secondary", use_container_width=True):
            st.session_state.excluded_clips.extend(candidates)
            new_candidates = find_top_3_highlights(
                video_path,
                duration,
                exclude_timestamps=st.session_state.excluded_clips
            )
            if not new_candidates:
                st.warning("已循环所有片段，重新开始。")
                st.session_state.excluded_clips = []
                new_candidates = find_top_3_highlights(video_path, duration, [])

            st.session_state.candidates = new_candidates
            st.rerun()

    with col_act2:
        # 生成：Primary 样式，填满宽度
        if st.button(t["btn_generate"], type="primary", use_container_width=True):
            clips_to_close = []
            try:
                start_time = candidates[selected_option]
                clip = VideoFileClip(video_path)
                clips_to_close.append(clip)

                # --- 布局引擎 ---
                orig_w, orig_h = clip.w, clip.h
                scale = 1.0

                min_side = min(orig_w, orig_h)
                if min_side < MIN_DIMENSION_PX:
                    scale = MIN_DIMENSION_PX / min_side
                elif max(orig_w, orig_h) > MAX_DIMENSION_PX:
                    scale = MAX_DIMENSION_PX / max(orig_w, orig_h)

                canvas_w, canvas_h = int(orig_w * scale), int(orig_h * scale)

                ratio = canvas_w / canvas_h
                if ratio > 1.2:
                    layout_mode = "LANDSCAPE"
                elif ratio < 0.8:
                    layout_mode = "PORTRAIT"
                else:
                    layout_mode = "SQUARE"

                padding, margin_side = 15, 20
                max_total_width = canvas_w - (margin_side * 2)
                h_by_height = int(canvas_h * 0.12)
                h_by_width = int((max_total_width - padding) / 6.5)
                final_btn_h = min(h_by_height, h_by_width)

                btn1_path = get_image_path(btn1_up, DEFAULT_BTN1_NAME, "AppStore", final_btn_h)
                btn2_path = get_image_path(btn2_up, DEFAULT_BTN2_NAME, "GooglePlay", final_btn_h)

                pil_b1 = trim_transparent_borders(btn1_path)
                btn1_clip = ImageClip(np.array(pil_b1)).resize(height=final_btn_h).set_duration(duration)
                pil_b2 = trim_transparent_borders(btn2_path)
                btn2_clip = ImageClip(np.array(pil_b2)).resize(height=final_btn_h).set_duration(duration)

                foreground = clip.subclip(start_time, start_time + duration)
                bg_frame = clip.get_frame(find_colorful_timestamp(video_path))
                background = create_blurred_background(bg_frame, (canvas_w, canvas_h)).set_duration(duration)

                scale_viz = st.session_state.scale_factor
                if layout_mode == "LANDSCAPE":
                    fg_w, fg_pos = int(canvas_w * 0.9 * scale_viz), ("center", int(canvas_h * 0.1))
                elif layout_mode == "PORTRAIT":
                    fg_w, fg_pos = int(canvas_w * 0.9 * scale_viz), ("center", int(canvas_h * 0.05))
                else:
                    fg_w, fg_pos = int(canvas_w * 0.8 * scale_viz), ("center", "center")

                foreground = foreground.resize(width=fg_w)
                foreground = foreground.margin(2, color=(255, 255, 255)).margin(bottom=4, right=4, color=(0, 0, 0),
                                                                                opacity=0.3)
                foreground = foreground.set_position(fg_pos)

                layers = [background, foreground]
                w1, w2 = btn1_clip.w, btn2_clip.w
                total_btn_w = w1 + w2 + padding
                start_x = (canvas_w - total_btn_w) / 2
                bottom_margin = 0.08 if layout_mode == "PORTRAIT" else 0.05
                btn_y = canvas_h - final_btn_h - int(canvas_h * bottom_margin)

                btn1_clip = btn1_clip.set_position((start_x, btn_y))
                btn2_clip = btn2_clip.set_position((start_x + w1 + padding, btn_y))
                layers.append(btn1_clip)
                layers.append(btn2_clip)

                if extra_img_up:
                    sticker_path = save_upload_to_temp(extra_img_up, "sticker")
                    if sticker_path:
                        pil_sticker = trim_transparent_borders(sticker_path)
                        sticker = ImageClip(np.array(pil_sticker)).set_duration(duration)
                        s_w = int(canvas_w * 0.2)
                        sticker = sticker.resize(width=s_w)
                        sticker = add_pulse_effect(sticker)
                        sticker = sticker.set_position((canvas_w - s_w - 10, 10))
                        layers.append(sticker)

                final_comp = CompositeVideoClip(layers, size=(canvas_w, canvas_h))
                out_path = os.path.join(tempfile.gettempdir(), f"final_{int(time.time())}.gif")

                success = smart_export_gif(final_comp, out_path, max_mb=MAX_FILE_SIZE_MB)

                if success or os.path.exists(out_path):
                    show_gif_robust(out_path)
                    with open(out_path, "rb") as f:
                        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                        with col_dl2:
                            st.download_button(
                                label=f"{t['download']} ({os.path.getsize(out_path) / 1024 / 1024:.2f}MB)",
                                data=f,
                                file_name="endcard.gif",
                                mime="image/gif",
                                type="primary",
                                use_container_width=True
                            )

                final_comp.close()
                btn1_clip.close()
                btn2_clip.close()

            except Exception as e:
                st.error(t["err_fail"].format(e=e))
            finally:
                for c in clips_to_close: c.close()

    st.markdown("---")

    # [修改] 底部重置区域：等宽、简洁、大方
    c_res1, c_res2 = st.columns(2)

    with c_res1:
        if st.button(t["btn_restart"], type="secondary", use_container_width=True):
            st.session_state.step = 1
            st.session_state.excluded_clips = []
            st.rerun()

    with c_res2:
        if st.button(t["btn_clear_reset"], type="secondary", use_container_width=True):
            st.session_state.step = 1
            st.session_state.temp_video_path = None  # 清除视频
            st.session_state.excluded_clips = []
            st.session_state.upload_key = str(time.time())  # 强制刷新上传组件
            st.rerun()
