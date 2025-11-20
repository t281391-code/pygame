"""
🚀 FACE LOCK + 2D GAME – Нэгтгэсэн код

- Tkinter дээр нүүр танилт (OpenCV, Deep Features)
- Танилт амжилттай болмогц pygame 2D тоглоом эхэлнэ
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import pickle
import time
from datetime import datetime
import numpy as np
import os
from collections import Counter
import threading
import json
import platform
import sys

# ==== GAME IMPORTS ====
import pygame
import pytmx
from pytmx.util_pygame import load_pygame
import random
import math
from enum import Enum


# ======================================================================
#                      FACE RECOGNITION (Tkinter)
# ======================================================================

class EnhancedFaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Face Lock + Game Launcher")
        self.root.geometry("1200x700")

        # OS
        self.is_macos = platform.system() == 'Darwin'
        self.is_windows = platform.system() == 'Windows'

        # Colors
        self.bg_dark = '#0a0e27'
        self.bg_panel = '#1a1f3a'
        self.fg_primary = '#00ff9f'
        self.fg_secondary = '#ffffff'
        self.fg_muted = '#666699'
        self.root.configure(bg=self.bg_dark)

        # Face data
        self.known_face_features = []
        self.known_face_names = []
        self.face_quality_scores = []
        self.data_file = "enhanced_face_data.pkl"
        self.threshold = 0.65

        # OpenCV
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml')

        # Video
        self.video_capture = None
        self.is_capturing = False
        self.current_mode = None
        self.fps = 0

        # Game unlock flags
        self.face_recognized = False
        self.recognized_name = None

        self.setup_ui()
        self.load_data_silent()

        # ПРОГРАМ ЭХЛЭХЭД АВТОМААТАР ТАНИЛТ ЭХЛҮҮЛНЭ
        self.root.after(800, self.auto_start_recognition)

    # ---------- UI / Helper ----------

    def auto_start_recognition(self):
        """Програм асахад автоматаар танилт эхлүүлэх"""
        # Хэрэв дата байвал шууд танилт эхлүүлнэ
        if self.known_face_names:
            self.start_recognition()
        else:
            self.update_status(
                "⚠️ Дата байхгүй, эхлээд 'Нүүр бүртгэх' ашиглан бүртгэнэ үү.")

    def get_font(self, size, weight='normal'):
        if self.is_macos:
            try:
                if weight == 'bold':
                    return ('SF Pro Display', size, 'bold')
                else:
                    return ('SF Pro Text', size)
            except:
                if weight == 'bold':
                    return ('Helvetica Neue', size, 'bold')
                else:
                    return ('Helvetica Neue', size)
        elif self.is_windows:
            if weight == 'bold':
                return ('Segoe UI', size, 'bold')
            else:
                return ('Segoe UI', size)
        else:
            return ('DejaVu Sans', size, weight)

    def setup_ui(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg=self.bg_panel, height=90)
        title_frame.pack(fill='x', pady=(0, 10))

        title_label = tk.Label(
            title_frame,
            text="🚀 FACE LOCK - RPG GAME LAUNCHER",
            font=self.get_font(26, 'bold'),
            bg=self.bg_panel,
            fg=self.fg_primary
        )
        title_label.pack(pady=15)

        mode_label = tk.Label(
            title_frame,
            text="🟢 Нүүр танилт амжилттай болсны дараа тоглоом эхэлнэ",
            font=self.get_font(10, 'bold'),
            bg=self.bg_panel,
            fg=self.fg_primary
        )
        mode_label.pack()

        # Main container
        main_container = tk.Frame(self.root, bg=self.bg_dark)
        main_container.pack(fill='both', expand=True, padx=20, pady=10)

        # Left panel
        left_panel = tk.Frame(main_container, bg=self.bg_panel, width=380)
        left_panel.pack(side='left', fill='both', padx=(0, 10))

        # Control buttons
        control_frame = tk.LabelFrame(
            left_panel,
            text="⚡ Үндсэн үйлдлүүд",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel,
            fg=self.fg_secondary,
            padx=15,
            pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        control_frame.pack(fill='x', pady=10, padx=10)

        self.register_btn = self.create_button(
            control_frame, "🤖 Нүүр бүртгэх", self.start_registration, '#00ff9f')
        if self.is_macos and hasattr(self.register_btn, '_frame'):
            self.register_btn._frame.pack(fill='x', pady=5)
        else:
            self.register_btn.pack(fill='x', pady=5)

        self.recognize_btn = self.create_button(
            control_frame, "🎥 Танилт эхлүүлэх", self.start_recognition, '#00aaff')
        if self.is_macos and hasattr(self.recognize_btn, '_frame'):
            self.recognize_btn._frame.pack(fill='x', pady=5)
        else:
            self.recognize_btn.pack(fill='x', pady=5)

        self.stop_btn = self.create_button(
            control_frame, "⏹️ Зогсоох", self.stop_capture, '#ff4466')
        if self.is_macos and hasattr(self.stop_btn, '_frame'):
            self.stop_btn._frame.pack(fill='x', pady=5)
        else:
            self.stop_btn.pack(fill='x', pady=5)
        self.stop_btn.config(state='disabled')

        # GAME BUTTON – танигдсаны дараа идэвхжиж тоглоом эхлүүлнэ
        self.game_btn = self.create_button(
            control_frame, "🎮 Тоглоом эхлүүлэх", self.launch_game_from_button, '#ff9500')
        if self.is_macos and hasattr(self.game_btn, '_frame'):
            self.game_btn._frame.pack(fill='x', pady=5)
        else:
            self.game_btn.pack(fill='x', pady=5)
        self.game_btn.config(state='disabled')

        # Advanced settings
        advanced_frame = tk.LabelFrame(
            left_panel,
            text="🎛️ Нарийвчилсан тохиргоо",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel,
            fg=self.fg_secondary,
            padx=15,
            pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        advanced_frame.pack(fill='x', pady=10, padx=10)

        checkbutton_bg = self.bg_panel
        checkbutton_fg = self.fg_secondary
        checkbutton_select = '#2a2f4a' if not self.is_macos else '#3a3f5a'

        self.multi_angle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="📐 Олон өнцгөөс авах",
            variable=self.multi_angle_var, bg=checkbutton_bg, fg=checkbutton_fg,
            selectcolor=checkbutton_select, font=self.get_font(10),
            activebackground=checkbutton_bg, activeforeground=checkbutton_fg
        ).pack(anchor='w', pady=3)

        self.quality_filter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="✨ Чанарын шүүлтүүр",
            variable=self.quality_filter_var, bg=checkbutton_bg, fg=checkbutton_fg,
            selectcolor=checkbutton_select, font=self.get_font(10),
            activebackground=checkbutton_bg, activeforeground=checkbutton_fg
        ).pack(anchor='w', pady=3)

        self.deep_features_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="🧠 Deep features (LBP+HOG+ORB)",
            variable=self.deep_features_var, bg=checkbutton_bg, fg=checkbutton_fg,
            selectcolor=checkbutton_select, font=self.get_font(10),
            activebackground=checkbutton_bg, activeforeground=checkbutton_fg
        ).pack(anchor='w', pady=3)

        self.show_confidence_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            advanced_frame, text="📊 Confidence bar",
            variable=self.show_confidence_var, bg=checkbutton_bg, fg=checkbutton_fg,
            selectcolor=checkbutton_select, font=self.get_font(10),
            activebackground=checkbutton_bg, activeforeground=checkbutton_fg
        ).pack(anchor='w', pady=3)

        # Data management
        data_frame = tk.LabelFrame(
            left_panel, text="💾 Дата удирдлага",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel, fg=self.fg_secondary, padx=15, pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        data_frame.pack(fill='x', pady=10, padx=10)

        buttons = [
            ("📂 Дата ачаалах", self.load_data, '#9966ff'),
            ("💾 Дата хадгалах", self.save_data, '#9966ff'),
            ("📤 Export JSON", self.export_json, '#ff9500'),
            ("📥 Import зураг", self.import_from_folder, '#ff9500'),
            ("👥 Хүмүүсийг харах", self.show_people_list, '#00aaff'),
            ("📈 Статистик", self.show_statistics, '#00aaff'),
            ("🗑️ Хүн устгах", self.delete_person, '#ff4466'),
        ]
        for text, cmd, color in buttons:
            btn = self.create_button(data_frame, text, cmd, color)
            if self.is_macos and hasattr(btn, '_frame'):
                btn._frame.pack(fill='x', pady=3)
            else:
                btn.pack(fill='x', pady=3)

        # Settings
        settings_frame = tk.LabelFrame(
            left_panel, text="⚙️ Тохиргоо",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel, fg=self.fg_secondary, padx=15, pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        settings_frame.pack(fill='x', pady=10, padx=10)

        tk.Label(settings_frame, text="Threshold утга:",
                 bg=self.bg_panel, fg=self.fg_secondary, font=self.get_font(10)).pack(anchor='w')

        self.threshold_var = tk.DoubleVar(value=self.threshold)
        threshold_slider = ttk.Scale(
            settings_frame, from_=0.50, to=0.85,
            variable=self.threshold_var, orient='horizontal',
            command=self.update_threshold
        )
        threshold_slider.pack(fill='x', pady=5)

        self.threshold_label = tk.Label(
            settings_frame, text=f"Утга: {self.threshold:.2f}",
            bg=self.bg_panel, fg=self.fg_primary, font=self.get_font(9, 'bold')
        )
        self.threshold_label.pack()

        # Status display
        status_frame = tk.LabelFrame(
            left_panel, text="📊 Систем мэдээлэл",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel, fg=self.fg_secondary, padx=15, pady=15,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        status_frame.pack(fill='both', expand=True, pady=10, padx=10)

        monospace_font = 'Menlo' if self.is_macos else (
            'Consolas' if self.is_windows else 'Monaco')
        self.status_text = tk.Text(
            status_frame, height=12, bg=self.bg_dark, fg=self.fg_primary,
            font=(monospace_font, 9), wrap='word', state='disabled',
            borderwidth=0, highlightthickness=0,
            insertbackground=self.fg_primary
        )
        self.status_text.pack(fill='both', expand=True)

        # Right panel - video
        right_panel = tk.Frame(main_container, bg=self.bg_panel)
        right_panel.pack(side='right', fill='both', expand=True)

        video_frame = tk.LabelFrame(
            right_panel, text="📹 Видео харагдац",
            font=self.get_font(12, 'bold'),
            bg=self.bg_panel, fg=self.fg_secondary,
            relief='flat' if self.is_macos else 'groove',
            borderwidth=1
        )
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.video_label = tk.Label(
            video_frame, bg=self.bg_dark,
            text="🎥 Видео зогссон байна\n\n✨ Танилт автоматаар эхлэх болно...",
            font=self.get_font(14), fg=self.fg_muted
        )
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)

        # Info bar
        info_frame = tk.Frame(right_panel, bg=self.bg_panel, height=40)
        info_frame.pack(fill='x', padx=10, pady=(0, 10))

        self.info_label = tk.Label(
            info_frame, text="⚡ Бэлэн",
            font=self.get_font(10), bg=self.bg_panel, fg=self.fg_primary
        )
        self.info_label.pack(side='left', padx=10, pady=5)

        self.update_status_display()

    def create_button(self, parent, text, command, color):
        if self.is_macos:
            btn_frame = tk.Frame(
                parent, bg=color, relief='flat', borderwidth=0)
            btn = tk.Button(
                btn_frame, text=text, command=command,
                bg=color, fg='#ffffff', font=self.get_font(11, 'bold'),
                relief='flat', cursor='hand2', height=2,
                activebackground=self.lighten_color(color),
                activeforeground='#ffffff',
                borderwidth=0, highlightthickness=0,
                highlightbackground=color,
                highlightcolor=color
            )
            btn.pack(fill='both', expand=True)

            def on_enter(e):
                btn_frame.config(bg=self.lighten_color(color))
                btn.config(bg=self.lighten_color(color),
                           activebackground=self.lighten_color(color),
                           highlightbackground=self.lighten_color(color))

            def on_leave(e):
                btn_frame.config(bg=color)
                btn.config(bg=color,
                           activebackground=self.lighten_color(color),
                           highlightbackground=color)

            btn_frame.bind('<Enter>', on_enter)
            btn_frame.bind('<Leave>', on_leave)
            btn.bind('<Enter>', on_enter)
            btn.bind('<Leave>', on_leave)
            btn._frame = btn_frame
            return btn
        else:
            btn = tk.Button(
                parent, text=text, command=command,
                bg=color, fg='#ffffff', font=self.get_font(11, 'bold'),
                relief='flat', cursor='hand2', height=2,
                activebackground=self.lighten_color(color),
                activeforeground='#ffffff',
                borderwidth=0, highlightthickness=0
            )
            btn.bind('<Enter>', lambda e: btn.config(
                bg=self.lighten_color(color)))
            btn.bind('<Leave>', lambda e: btn.config(bg=color))
            return btn

    def lighten_color(self, color):
        colors = {
            '#00ff9f': '#33ffb3', '#00aaff': '#33bbff',
            '#ff4466': '#ff6688', '#9966ff': '#aa77ff',
            '#ff9500': '#ffaa33'
        }
        return colors.get(color, color)

    # ---------- Feature extraction / comparison (same as your code) ----------

    def extract_deep_features(self, face_image):
        try:
            if face_image is None or face_image.size == 0:
                return None

            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image

            if gray.shape[0] < 20 or gray.shape[1] < 20:
                return None

            gray = cv2.resize(gray, (128, 128))
            gray = cv2.equalizeHist(gray)

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()[:64]

            lbp = self.compute_lbp(gray)
            if lbp is not None and lbp.size > 0:
                lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
                lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()[:64]
            else:
                lbp_hist = np.zeros(64)

            hog = self.compute_hog(gray)
            if len(hog) > 128:
                hog = hog[:128]
            elif len(hog) < 128:
                hog = np.pad(hog, (0, 128 - len(hog)), 'constant')

            orb_feat = self.compute_orb_features(gray)
            if len(orb_feat) < 32:
                orb_feat = np.pad(
                    orb_feat, (0, 32 - len(orb_feat)), 'constant')

            combined = np.concatenate([hist, lbp_hist, hog, orb_feat])

            if np.any(np.isnan(combined)) or np.any(np.isinf(combined)):
                return None

            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            else:
                return None

            return combined.astype(np.float32)
        except Exception:
            return None

    def compute_lbp(self, image):
        height, width = image.shape
        lbp = np.zeros((height-2, width-2), dtype=np.uint8)
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        return lbp

    def compute_hog(self, image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        bins = np.int32(angle / 40) % 9
        hist = []
        cell_size = 16
        for i in range(0, image.shape[0] - cell_size, cell_size):
            for j in range(0, image.shape[1] - cell_size, cell_size):
                cell_mag = mag[i:i+cell_size, j:j+cell_size]
                cell_bins = bins[i:i+cell_size, j:j+cell_size]
                cell_hist = np.zeros(9)
                for k in range(9):
                    cell_hist[k] = np.sum(cell_mag[cell_bins == k])
                hist.extend(cell_hist)
        hog_features = np.array(hist)
        if np.linalg.norm(hog_features) > 0:
            hog_features = hog_features / np.linalg.norm(hog_features)
        return hog_features

    def compute_orb_features(self, image):
        try:
            orb = cv2.ORB_create(nfeatures=50)
            keypoints, descriptors = orb.detectAndCompute(image, None)
            if descriptors is not None and len(descriptors) > 0:
                avg_desc = np.mean(descriptors, axis=0)
                if len(avg_desc) > 32:
                    return avg_desc[:32]
                else:
                    padded = np.zeros(32)
                    padded[:len(avg_desc)] = avg_desc
                    return padded
            else:
                return np.zeros(32)
        except:
            return np.zeros(32)

    def calculate_face_quality(self, face_image):
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(
                face_image.shape) == 3 else face_image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            brightness = np.mean(gray)
            brightness_score = 100 - abs(brightness - 128)
            contrast = gray.std()
            quality = min(
                100, (sharpness * 3 + brightness_score + contrast) / 5)
            return max(0, quality)
        except:
            return 50.0

    def compare_features(self, feat1, feat2):
        try:
            feat1 = np.array(feat1, dtype=np.float32)
            feat2 = np.array(feat2, dtype=np.float32)
            if np.any(np.isnan(feat1)) or np.any(np.isnan(feat2)):
                return 0.0
            if np.any(np.isinf(feat1)) or np.any(np.isinf(feat2)):
                return 0.0
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            feat1_norm = feat1 / norm1
            feat2_norm = feat2 / norm2
            cos_sim = np.clip(np.dot(feat1_norm, feat2_norm), -1.0, 1.0)
            euclidean_dist = np.linalg.norm(feat1_norm - feat2_norm)
            euclidean_sim = 1 / (1 + euclidean_dist)
            similarity = 0.7 * cos_sim + 0.3 * euclidean_sim
            return np.clip(similarity, 0.0, 1.0)
        except Exception:
            return 0.0

    # ---------- Registration / Recognition ----------

    def start_registration(self):
        if self.is_capturing:
            messagebox.showwarning("Анхааруулга", "Өөр үйлдэл явагдаж байна!")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("✨ Нүүр бүртгэх")
        dialog.geometry("450x280")
        dialog.configure(bg=self.bg_panel)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="👤 Хүний нэр оруулна уу:",
                 font=self.get_font(13, 'bold'),
                 bg=self.bg_panel, fg=self.fg_secondary).pack(pady=25)

        name_entry = tk.Entry(dialog, font=self.get_font(12), width=30,
                              bg='#2a2f4a', fg=self.fg_secondary,
                              insertbackground=self.fg_primary, relief='flat', borderwidth=5)
        name_entry.pack(pady=10)
        name_entry.focus()

        tk.Label(dialog, text="📸 Зургийн тоо:",
                 font=self.get_font(10),
                 bg=self.bg_panel, fg=self.fg_secondary).pack(pady=(10, 5))

        sample_var = tk.IntVar(value=10)
        tk.Spinbox(dialog, from_=6, to=20, textvariable=sample_var,
                   font=self.get_font(11), width=10,
                   bg='#2a2f4a', fg=self.fg_secondary).pack()

        def submit():
            name = name_entry.get().strip()
            if name:
                dialog.destroy()
                self.register_name = name
                self.register_samples = sample_var.get()
                threading.Thread(target=self.register_thread,
                                 daemon=True).start()
            else:
                messagebox.showerror("Алдаа", "Нэр оруулна уу!")

        submit_btn = tk.Button(dialog, text="✓ Эхлүүлэх", command=submit,
                               bg=self.fg_primary, fg=self.bg_dark,
                               font=self.get_font(11, 'bold'),
                               cursor='hand2', height=2, relief='flat',
                               activebackground=self.lighten_color(
                                   self.fg_primary),
                               activeforeground=self.bg_dark)
        submit_btn.pack(pady=15)
        name_entry.bind('<Return>', lambda e: submit())

    def is_face_centered(self, face_rect, frame_shape, center_threshold=0.15):
        x, y, w, h = face_rect
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        dx = abs(face_center_x - frame_center_x) / frame_shape[1]
        dy = abs(face_center_y - frame_center_y) / frame_shape[0]
        return dx < center_threshold and dy < center_threshold

    def draw_center_guide(self, frame):
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        line_length = 30
        thickness = 2
        color = (100, 100, 255)
        cv2.line(frame, (center_x - line_length, center_y),
                 (center_x + line_length, center_y), color, thickness)
        cv2.line(frame, (center_x, center_y - line_length),
                 (center_x, center_y + line_length), color, thickness)
        cv2.circle(frame, (center_x, center_y), 50, color, 2)

    def clean_features(self, features_list):
        cleaned_features = []
        for features in features_list:
            try:
                features_array = np.array(features, dtype=np.float32)
                if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                    continue
                norm = np.linalg.norm(features_array)
                if norm > 0:
                    features_array = features_array / norm
                else:
                    continue
                cleaned_features.append(features_array)
            except:
                continue
        return cleaned_features

    def register_thread(self):
        self.is_capturing = True
        self.current_mode = 'register'
        self.register_btn.config(state='disabled')
        self.recognize_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        self.update_status(f"\n🚀 {self.register_name} бүртгэж байна...")
        self.update_status("📍 Нүүрээ камерын төвд байрлуулна уу")
        self.info_label.config(text=f"📸 Бүртгэл: {self.register_name}")

        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)

        features_list = []
        quality_list = []
        count = 0
        face_positions = []
        last_capture = time.time()
        stable_frames = 0
        centered_frames = 0

        process_interval = 1.0 / 10
        last_process_time = time.time()
        last_faces = []

        while count < self.register_samples and self.is_capturing:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            self.draw_center_guide(frame)
            current_time = time.time()

            if current_time - last_process_time >= process_interval:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(120, 120), maxSize=(400, 400)
                )
                last_faces = faces
                last_process_time = current_time
            else:
                faces = last_faces

            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, minNeighbors=8)
                has_eyes = len(eyes) >= 2
                face_center = (x + w//2, y + h//2)
                is_centered = self.is_face_centered((x, y, w, h), frame.shape)
                is_new_angle = self.is_new_angle(
                    face_center, face_positions) if self.multi_angle_var.get() else True
                face_roi = frame[y:y+h, x:x+w]
                quality = self.calculate_face_quality(face_roi)
                quality_ok = quality > 35 if self.quality_filter_var.get() else True
                ready = has_eyes and is_centered and is_new_angle and quality_ok

                if ready:
                    color = (0, 255, 0)
                    stable_frames += 1
                    centered_frames += 1
                else:
                    if not is_centered:
                        color = (0, 165, 255)
                        centered_frames = 0
                    elif not quality_ok:
                        color = (255, 0, 255)
                        centered_frames = 0
                    elif not has_eyes:
                        color = (0, 255, 255)
                        centered_frames = 0
                    else:
                        color = (0, 165, 255)
                        centered_frames = 0
                    stable_frames = 0

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                if is_centered:
                    cv2.circle(frame, face_center, 5, (0, 255, 0), -1)

                status_text = []
                if self.quality_filter_var.get():
                    status_text.append(f"Q: {quality:.0f}%")
                if not is_centered:
                    status_text.append("Центрт байрлуулна уу")
                elif not has_eyes:
                    status_text.append("Нүд харагдахгүй")
                elif ready:
                    status_text.append("✓ Бэлэн")

                if status_text:
                    text = " | ".join(status_text)
                    cv2.putText(frame, text, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if ready and stable_frames >= 3 and centered_frames >= 2 and current_time - last_capture >= 0.4:
                    if self.deep_features_var.get():
                        features = self.extract_deep_features(face_roi)
                    else:
                        features = self.extract_simple_features(face_roi)
                    if features is not None and len(features) > 0:
                        try:
                            features_array = np.array(
                                features, dtype=np.float32)
                            if not np.any(np.isnan(features_array)) and not np.any(np.isinf(features_array)):
                                norm = np.linalg.norm(features_array)
                                if norm > 0:
                                    features_array = features_array / norm
                                    features_list.append(features_array)
                                    quality_list.append(quality)
                                    face_positions.append(face_center)
                                    count += 1
                                    last_capture = current_time
                                    stable_frames = 0
                                    centered_frames = 0
                                    overlay = frame.copy()
                                    cv2.circle(
                                        overlay, (frame.shape[1]//2, frame.shape[0]//2), 100, (0, 255, 0), -1)
                                    frame = cv2.addWeighted(
                                        frame, 0.6, overlay, 0.4, 0)
                                    self.update_status(
                                        f"📸 {count}/{self.register_samples} - Q: {quality:.0f}%")
                        except:
                            pass

            self.draw_progress(frame, count, self.register_samples)
            self.display_frame(frame)
            time.sleep(0.01)

        self.video_capture.release()

        if count >= 3:
            cleaned_data = []
            for i, features in enumerate(features_list):
                try:
                    features_array = np.array(features, dtype=np.float32)
                    if not np.any(np.isnan(features_array)) and not np.any(np.isinf(features_array)):
                        norm = np.linalg.norm(features_array)
                        if norm > 0:
                            features_array = features_array / norm
                            cleaned_data.append({
                                'features': features_array,
                                'quality': quality_list[i],
                                'position': face_positions[i]
                            })
                except:
                    continue

            if len(cleaned_data) >= 3:
                unique_features = []
                unique_qualities = []
                for data in cleaned_data:
                    feat = data['features']
                    is_duplicate = False
                    for existing_feat in unique_features:
                        similarity = self.compare_features(
                            feat, existing_feat)
                        if similarity > 0.95:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_features.append(feat)
                        unique_qualities.append(data['quality'])

                for i, features in enumerate(unique_features):
                    self.known_face_features.append(features)
                    self.known_face_names.append(self.register_name)
                    self.face_quality_scores.append(unique_qualities[i])

                avg_quality = np.mean(unique_qualities)
                self.update_status(
                    f"✅ {self.register_name} амжилттай бүртгэгдлээ!")
                self.update_status(f"📊 Дундаж чанар: {avg_quality:.1f}%")
                self.update_status(
                    f"🧹 Цэвэрлэсэн: {len(unique_features)}/{len(cleaned_data)} зураг")
                self.save_data()
                self.update_status_display()
            else:
                self.update_status(f"❌ Хангалттай цэвэр дата байхгүй!")
        else:
            self.update_status(f"❌ Хангалттай зураг аваагүй!")

        self.stop_capture()

    def extract_simple_features(self, face_image):
        try:
            if face_image is None or face_image.size == 0:
                return None
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(
                face_image.shape) == 3 else face_image
            if gray.shape[0] < 20 or gray.shape[1] < 20:
                return None
            gray = cv2.resize(gray, (100, 100))
            gray = cv2.equalizeHist(gray)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            features = cv2.normalize(hist, hist).flatten()
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return None
            return features.astype(np.float32)
        except:
            return None

    def is_new_angle(self, center, positions, min_diff=30):
        for pos in positions:
            dist = np.sqrt((center[0] - pos[0])**2 +
                           (center[1] - pos[1])**2)
            if dist < min_diff:
                return False
        return True

    def start_recognition(self):
        if not self.known_face_names:
            messagebox.showwarning("Анхааруулга", "Эхлээд дата ачаална уу!")
            return
        if self.is_capturing:
            messagebox.showwarning("Анхааруулга", "Өөр үйлдэл явагдаж байна!")
            return
        self.current_mode = 'recognize'
        self.is_capturing = True
        self.register_btn.config(state='disabled')
        self.recognize_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.update_status("\n🎥 AI танилт эхэллээ...")
        self.info_label.config(text="🔍 Танилт явагдаж байна...")
        threading.Thread(target=self.recognize_thread, daemon=True).start()

    def recognize_thread(self):
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        last_results = {}
        fps_start = time.time()
        fps_counter = 0
        last_process_time = time.time()
        process_interval = 1.0 / 10

        stable_name = None
        stable_frames = 0
        required_stable_frames = 15  # ийм олон фрэйм дараалан танигдвал "батлагдлаа"

        while self.is_capturing:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            current_time = time.time()
            frame_count += 1
            fps_counter += 1
            if current_time - fps_start >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_start = current_time

            if current_time - last_process_time >= process_interval:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5,
                    minSize=(60, 60), maxSize=(400, 400)
                )
                new_results = {}
                best_frame_name = None
                best_frame_conf = 0.0

                for face_id, (x, y, w, h) in enumerate(faces):
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                        continue
                    try:
                        if self.deep_features_var.get():
                            features = self.extract_deep_features(face_roi)
                        else:
                            features = self.extract_simple_features(face_roi)
                        if features is not None and len(features) > 0:
                            features_array = np.array(
                                features, dtype=np.float32)
                            if not np.any(np.isnan(features_array)) and not np.any(np.isinf(features_array)):
                                name, confidence = self.find_best_match(
                                    features)
                                new_results[face_id] = (
                                    x, y, w, h, name, confidence)
                                if name != "Танигдаагүй" and confidence > best_frame_conf:
                                    best_frame_name = name
                                    best_frame_conf = confidence
                    except Exception:
                        continue

                # stablize recognition
                if best_frame_name and best_frame_conf >= self.threshold * 100:
                    if stable_name == best_frame_name:
                        stable_frames += 1
                    else:
                        stable_name = best_frame_name
                        stable_frames = 1
                else:
                    stable_name = None
                    stable_frames = 0

                # if stable enough and not yet unlocked
                if (stable_name is not None and
                        stable_frames >= required_stable_frames and
                        not self.face_recognized):
                    self.face_recognized = True
                    self.recognized_name = stable_name
                    self.root.after(0, self.on_face_recognized)

                last_results = new_results
                last_process_time = current_time

            for face_id, (x, y, w, h, name, confidence) in last_results.items():
                color = self.get_color(name, confidence)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                corner_len = 20
                for (cx, cy) in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
                    dx = corner_len if cx == x else -corner_len
                    dy = corner_len if cy == y else -corner_len
                    cv2.line(frame, (cx, cy), (cx+dx, cy), color, 5)
                    cv2.line(frame, (cx, cy), (cx, cy+dy), color, 5)

                if self.show_confidence_var.get() and name != "Танигдаагүй":
                    bar_width = int(w * (confidence / 100))
                    cv2.rectangle(frame, (x, y-10),
                                  (x+bar_width, y-5), color, -1)

                label_text = f"{name} ({confidence:.0f}%)" if name != "Танигдаагүй" else name
                label_y = y - 15 if y - 15 > 15 else y + h + 35
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame, (x, label_y - text_height - 10),
                              (x + text_width + 10, label_y), color, -1)
                cv2.putText(frame, label_text, (x + 5, label_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # HUD + unlock progress
            self.draw_hud(frame, len(last_results))
            if not self.face_recognized:
                progress = min(
                    100, (stable_frames / required_stable_frames) * 100)
                cv2.putText(frame, f"Face Lock: {int(progress)}%",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 0), 2)
            else:
                cv2.putText(frame, f"UNLOCKED: {self.recognized_name}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

            self.display_frame(frame)
            time.sleep(0.03)

        self.video_capture.release()
        self.stop_capture()

    def on_face_recognized(self):
        """Нүүр амжилттай танигдмагц дуудагдана"""
        self.update_status(
            f"\n✅ Нүүр амжилттай танигдлаа: {self.recognized_name}")
        self.info_label.config(
            text=f"✅ Танигдсан: {self.recognized_name} - Тоглоом эхлүүлэх боломжтой!")
        self.game_btn.config(state='normal')

        # Шууд асууж, шууд тоглоом эхлүүлэх
        if messagebox.askyesno(
            "Face Lock",
            f"{self.recognized_name} танигдлаа.\n\nТоглоомыг эхлүүлэх үү?"
        ):
            self.stop_capture()
            self.launch_game()

    def find_best_match(self, features):
        if not self.known_face_features:
            return "Танигдаагүй", 0
        if features is None:
            return "Танигдаагүй", 0
        try:
            features = np.array(features, dtype=np.float32)
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return "Танигдаагүй", 0
            max_similarity = 0
            best_name = "Танигдаагүй"
            for idx, known_features in enumerate(self.known_face_features):
                if known_features is None:
                    continue
                similarity = self.compare_features(features, known_features)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_name = self.known_face_names[idx]
            if max_similarity >= self.threshold:
                confidence = max_similarity * 100
                return best_name, confidence
            else:
                return "Танигдаагүй", max_similarity * 100
        except Exception:
            return "Танигдаагүй", 0

    def get_color(self, name, confidence):
        if name != "Танигдаагүй":
            if confidence > 80:
                return (0, 255, 159)
            elif confidence > 70:
                return (0, 191, 255)
            else:
                return (0, 165, 255)
        return (0, 0, 255)

    def draw_hud(self, frame, face_count):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (26, 31, 58), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        info = f"FPS: {self.fps} | Faces: {face_count} | Enhanced OpenCV"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 159), 2)
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_progress(self, frame, current, total):
        h, w = frame.shape[:2]
        bar_width = w - 80
        bar_height = 35
        bar_x, bar_y = 40, h - 60
        cv2.rectangle(frame, (bar_x-5, bar_y-5),
                      (bar_x + bar_width + 5, bar_y + bar_height + 5),
                      (26, 31, 58), -1)
        progress = int((current / total) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + progress, bar_y + bar_height),
                      (0, 255, 159), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (255, 255, 255), 2)
        text = f"{current}/{total} ({int(current/total*100)}%)"
        cv2.putText(frame, text, (bar_x + bar_width//2 - 60, bar_y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def stop_capture(self):
        self.is_capturing = False
        if self.video_capture:
            self.video_capture.release()
        self.register_btn.config(state='normal')
        self.recognize_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.info_label.config(text="⚡ Бэлэн")
        self.video_label.config(
            image='',
            text="🎥 Видео зогссон байна\n\n✨ Танилтыг дахин эхлүүлэх бол товчийг дарна уу"
        )

    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((900, 650), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk, text='')

    def update_status(self, message, clear=False):
        self.status_text.config(state='normal')
        if clear:
            self.status_text.delete(1.0, tk.END)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')

    def update_status_display(self):
        self.update_status("", clear=True)
        self.update_status("🟢 Enhanced OpenCV Mode (Face Lock)")
        if self.known_face_names:
            name_counts = Counter(self.known_face_names)
            self.update_status(f"\n👥 Бүртгэлтэй: {len(name_counts)} хүн")
            self.update_status(f"📊 Нийт зураг: {len(self.known_face_names)}")
            if self.face_quality_scores:
                avg_quality = np.mean(self.face_quality_scores)
                self.update_status(f"✨ Дундаж чанар: {avg_quality:.1f}%")
            self.update_status(f"🎯 Threshold: {self.threshold:.2f}\n")
            self.update_status("📋 Хүмүүс:")
            for name, count in sorted(name_counts.items()):
                self.update_status(f"  • {name}: {count} зураг")
        else:
            self.update_status("\n⚠️ Бүртгэлтэй хүн байхгүй")

    def update_threshold(self, value):
        self.threshold = float(value)
        self.threshold_label.config(text=f"Утга: {self.threshold:.2f}")

    def save_data(self):
        if not self.known_face_names:
            messagebox.showwarning("Анхааруулга", "Хадгалах дата байхгүй!")
            return
        if os.path.exists(self.data_file):
            name_counts = Counter(self.known_face_names)
            total_people = len(name_counts)
            total_samples = len(self.known_face_names)
            people_list = "\n".join(
                [f"  • {name}: {count} зураг" for name, count in sorted(name_counts.items())])
            message = f"📋 БҮРТГЭЛТЭЙ ХҮМҮҮС\n" + "="*40 + "\n\n"
            message += f"👥 Нийт хүн: {total_people}\n"
            message += f"📸 Нийт зураг: {total_samples}\n\n"
            message += "📋 Хүмүүс:\n" + people_list + "\n\n"
            message += "Хадгалах уу?"
            result = messagebox.askyesno("Бүртгэлтэй хүмүүс", message)
            if not result:
                return
        try:
            cleaned_features = []
            cleaned_names = []
            cleaned_qualities = []
            for i, features in enumerate(self.known_face_features):
                try:
                    features_array = np.array(features, dtype=np.float32)
                    if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                        continue
                    norm = np.linalg.norm(features_array)
                    if norm > 0:
                        features_array = features_array / norm
                        cleaned_features.append(features_array)
                        cleaned_names.append(self.known_face_names[i])
                        cleaned_qualities.append(self.face_quality_scores[i] if i < len(
                            self.face_quality_scores) else 50.0)
                except:
                    continue
            self.known_face_features = cleaned_features
            self.known_face_names = cleaned_names
            self.face_quality_scores = cleaned_qualities
            data = {
                'features': self.known_face_features,
                'names': self.known_face_names,
                'quality_scores': self.face_quality_scores,
                'threshold': self.threshold,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': '2.1-opencv-optimized'
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
            self.update_status("💾 Дата хадгалагдлаа!")
            self.update_status(f"🧹 Цэвэрлэсэн: {len(cleaned_features)} зураг")
            messagebox.showinfo(
                "Амжилт", f"Ачаалагдлаа!\n{len(set(self.known_face_names))} хүн\n{len(cleaned_features)} цэвэр зураг")
        except Exception as e:
            messagebox.showerror("Алдаа", f"Хадгалах алдаа: {e}")

    def load_data_silent(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    raw_features = data.get('features', [])
                    raw_names = data.get('names', [])
                    raw_qualities = data.get('quality_scores', [])
                    self.threshold = data.get('threshold', self.threshold)
                    try:
                        self.threshold_var.set(self.threshold)
                    except:
                        pass
                cleaned_features = []
                cleaned_names = []
                cleaned_qualities = []
                for i, features in enumerate(raw_features):
                    try:
                        features_array = np.array(features, dtype=np.float32)
                        if not np.any(np.isnan(features_array)) and not np.any(np.isinf(features_array)):
                            norm = np.linalg.norm(features_array)
                            if norm > 0:
                                features_array = features_array / norm
                                cleaned_features.append(features_array)
                                cleaned_names.append(
                                    raw_names[i] if i < len(raw_names) else "Unknown")
                                cleaned_qualities.append(
                                    raw_qualities[i] if i < len(raw_qualities) else 50.0)
                    except:
                        continue
                self.known_face_features = cleaned_features
                self.known_face_names = cleaned_names
                self.face_quality_scores = cleaned_qualities
            except:
                pass

    def load_data(self):
        if not os.path.exists(self.data_file):
            messagebox.showwarning("Анхааруулга", "Дата файл олдсонгүй!")
            return
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                raw_features = data.get('features', [])
                raw_names = data.get('names', [])
                raw_qualities = data.get('quality_scores', [])
                self.threshold = data.get('threshold', self.threshold)
                self.threshold_var.set(self.threshold)
            cleaned_features = []
            cleaned_names = []
            cleaned_qualities = []
            for i, features in enumerate(raw_features):
                try:
                    features_array = np.array(features, dtype=np.float32)
                    if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                        continue
                    norm = np.linalg.norm(features_array)
                    if norm > 0:
                        features_array = features_array / norm
                        cleaned_features.append(features_array)
                        cleaned_names.append(
                            raw_names[i] if i < len(raw_names) else "Unknown")
                        cleaned_qualities.append(
                            raw_qualities[i] if i < len(raw_qualities) else 50.0)
                except:
                    continue
            self.known_face_features = cleaned_features
            self.known_face_names = cleaned_names
            self.face_quality_scores = cleaned_qualities
            self.update_status_display()
            cleaned_count = len(cleaned_features)
            original_count = len(raw_features)
            if cleaned_count < original_count:
                messagebox.showinfo("Амжилт",
                                    f"Ачаалагдлаа!\n{len(set(cleaned_names))} хүн\n"
                                    f"🧹 Цэвэрлэсэн: {cleaned_count}/{original_count} зураг")
            else:
                messagebox.showinfo(
                    "Амжилт", f"Ачаалагдлаа!\n{len(set(cleaned_names))} хүн")
        except Exception as e:
            messagebox.showerror("Алдаа", f"Ачаалах алдаа: {e}")

    def export_json(self):
        if not self.known_face_names:
            messagebox.showwarning("Анхааруулга", "Экспорт хийх дата байхгүй!")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                name_counts = Counter(self.known_face_names)
                export_data = {
                    'people': [
                        {
                            'name': name,
                            'sample_count': count,
                            'avg_quality': float(np.mean([
                                self.face_quality_scores[i]
                                for i, n in enumerate(self.known_face_names) if n == name
                            ]))
                        }
                        for name, count in name_counts.items()
                    ],
                    'total_samples': len(self.known_face_names),
                    'threshold': self.threshold,
                    'export_date': datetime.now().isoformat()
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                self.update_status(f"📤 Export: {filename}")
                messagebox.showinfo("Амжилт", "JSON амжилттай!")
            except Exception as e:
                messagebox.showerror("Алдаа", f"Export алдаа: {e}")

    def import_from_folder(self):
        folder = filedialog.askdirectory(title="Зургийн фолдер сонгох")
        if folder:
            files = [f for f in os.listdir(folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not files:
                messagebox.showwarning("Анхааруулга", "Зураг олдсонгүй!")
                return
            success = 0
            for filename in files:
                path = os.path.join(folder, filename)
                image = cv2.imread(path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) > 0:
                    face = max(faces, key=lambda r: r[2] * r[3])
                    x, y, w, h = face
                    face_roi = image[y:y+h, x:x+w]
                    features = self.extract_deep_features(face_roi) if self.deep_features_var.get(
                    ) else self.extract_simple_features(face_roi)
                    if features is not None:
                        name = os.path.splitext(
                            filename)[0].replace('_', ' ').title()
                        quality = self.calculate_face_quality(face_roi)
                        self.known_face_features.append(features)
                        self.known_face_names.append(name)
                        self.face_quality_scores.append(quality)
                        success += 1
            if success > 0:
                self.save_data()
                self.update_status_display()
                messagebox.showinfo(
                    "Амжилт", f"{success}/{len(files)} зураг импортлогдлоо!")
            else:
                messagebox.showwarning(
                    "Анхааруулга", "Амжилттай импортлосон зураг байхгүй!")

    def show_people_list(self):
        if not self.known_face_names:
            messagebox.showinfo("Мэдээлэл", "Бүртгэлтэй хүн байхгүй")
            return
        name_counts = Counter(self.known_face_names)
        message = "📋 БҮРТГЭЛТЭЙ ХҮМҮҮС\n" + "="*40 + "\n\n"
        for name, count in sorted(name_counts.items()):
            qualities = [self.face_quality_scores[i]
                         for i, n in enumerate(self.known_face_names) if n == name]
            avg_q = np.mean(qualities) if qualities else 0
            message += f"👤 {name}\n"
            message += f"   📊 Зураг: {count}\n"
            message += f"   ✨ Чанар: {avg_q:.1f}%\n\n"
        messagebox.showinfo("Бүртгэлтэй хүмүүс", message)

    def show_statistics(self):
        if not self.known_face_names:
            messagebox.showinfo("Мэдээлэл", "Статистик байхгүй")
            return
        name_counts = Counter(self.known_face_names)
        total_people = len(name_counts)
        total_samples = len(self.known_face_names)
        avg_quality = np.mean(
            self.face_quality_scores) if self.face_quality_scores else 0
        message = "📊 СТАТИСТИК\n" + "="*40 + "\n\n"
        message += f"👥 Нийт хүн: {total_people}\n"
        message += f"📸 Нийт зураг: {total_samples}\n"
        message += f"✨ Дундаж чанар: {avg_quality:.1f}%\n"
        message += f"🎯 Threshold: {self.threshold:.2f}\n\n"
        message += "📈 Хүн бүрийн зураг:\n"
        for name, count in name_counts.most_common():
            message += f"  • {name}: {count}\n"
        messagebox.showinfo("Статистик", message)

    def delete_person(self):
        if not self.known_face_names:
            messagebox.showinfo("Мэдээлэл", "Бүртгэлтэй хүн байхгүй")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("🗑️ Хүн устгах")
        dialog.geometry("450x350")
        dialog.configure(bg=self.bg_panel)
        dialog.transient(self.root)
        dialog.grab_set()
        tk.Label(dialog, text="⚠️ Устгах хүнийг сонгоно уу:",
                 font=self.get_font(12, 'bold'),
                 bg=self.bg_panel, fg=self.fg_secondary).pack(pady=20)
        name_counts = Counter(self.known_face_names)
        names = sorted(name_counts.keys())
        listbox = tk.Listbox(dialog, font=self.get_font(11), height=10,
                             bg='#2a2f4a', fg=self.fg_secondary,
                             selectbackground=self.fg_primary,
                             selectforeground=self.bg_dark)
        listbox.pack(fill='both', expand=True, padx=20, pady=10)
        for name in names:
            listbox.insert(tk.END, f"{name} ({name_counts[name]} зураг)")

        def delete_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Анхааруулга", "Хүн сонгоно уу!")
                return
            name = names[selection[0]]
            confirm = messagebox.askyesno(
                "Баталгаажуулалт",
                f"'{name}' устгах уу?\n\nБуцаах боломжгүй!"
            )
            if confirm:
                indices = [i for i, n in enumerate(
                    self.known_face_names) if n == name]
                for idx in sorted(indices, reverse=True):
                    del self.known_face_features[idx]
                    del self.known_face_names[idx]
                    if idx < len(self.face_quality_scores):
                        del self.face_quality_scores[idx]
                self.update_status(f"🗑️ {name} устгагдлаа!")
                self.save_data()
                self.update_status_display()
                dialog.destroy()

        delete_btn = tk.Button(dialog, text="🗑️ Устгах", command=delete_selected,
                               bg='#ff4466', fg='#ffffff',
                               font=self.get_font(11, 'bold'),
                               cursor='hand2', height=2, relief='flat',
                               activebackground=self.lighten_color('#ff4466'),
                               activeforeground='#ffffff')
        delete_btn.pack(pady=10)

    # ---------- GAME LAUNCH INTEGRATION ----------

    def launch_game_from_button(self):
        """Game товчийг гараар дарахад"""
        if not self.face_recognized or not self.recognized_name:
            messagebox.showwarning(
                "Анхааруулга", "Эхлээд нүүр танилт хийгээд танигдах хэрэгтэй!")
            return
        self.launch_game()

    def launch_game(self):
        """Pygame тоглоомыг тусдаа thread дээр эхлүүлэх"""
        self.update_status("🎮 Тоглоом эхлэж байна...")
        self.info_label.config(text="🎮 Тоглоом эхэллээ (pygame)...")
        # Tk цонхыг багасгах (хүсвэл)
        # self.root.withdraw()
        threading.Thread(target=self.run_game_wrapper, daemon=True).start()

    def run_game_wrapper(self):
        """Actual game runner - pygame Game"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except:
            script_dir = os.getcwd()
        main_map_path = r"ТАНЫ_ФАЙЛЫН_БҮРЭН_ЗАМ\main_map.tmx"
        if not os.path.exists(main_map_path):
            print("main_map.tmx not found in the map folder!")
            messagebox.showerror(
                "Алдаа", "main_map.tmx олдсонгүй. 'map' фолдерт оруулна уу.")
            return
        # fullscreen=False байлгавал development-д амар
        game = Game(main_map_path, fullscreen=False)
        game.run()


# ======================================================================
#                          GAME CODE (pygame)
# ======================================================================

# --- Таны game-ийн бүх код яг хэвээр нь доор байна ---


class State(Enum):
    IDLE = 0
    ATTACKING = 1
    HURT = 2
    DEAD = 3


class Projectile:
    def __init__(self, x, y, target_x, target_y, damage, is_enemy=False, projectile_type='default'):
        self.x = x
        self.y = y
        self.damage = damage
        self.speed = 8
        self.is_enemy = is_enemy
        self.active = True
        self.projectile_type = projectile_type

        dx = target_x - x
        dy = target_y - y
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            self.vel_x = (dx / length) * self.speed
            self.vel_y = (dy / length) * self.speed
        else:
            self.vel_x = 0
            self.vel_y = 0

        try:
            base_dir = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'image')
            if projectile_type == 'fire':
                img_path = os.path.join(base_dir, 'fire_effect.png')
            elif projectile_type == 'water':
                img_path = os.path.join(base_dir, 'water_effect.png')
            elif projectile_type == 'void':
                img_path = os.path.join(base_dir, 'void_effect.png')
            elif projectile_type == 'ice':
                img_path = os.path.join(base_dir, 'ice_effect.png')
            elif projectile_type == 'lightning':
                img_path = os.path.join(base_dir, 'lightning_effect.png')
            elif projectile_type == 'holy':
                img_path = os.path.join(base_dir, 'holy_effect.png')
            else:
                img_path = os.path.join(base_dir, 'projectile.png')

            if os.path.exists(img_path):
                self.image = pygame.image.load(img_path).convert_alpha()
                self.image = pygame.transform.scale(self.image, (20, 20))
            else:
                self.image = pygame.Surface((20, 20), pygame.SRCALPHA)
                if projectile_type == 'fire':
                    pygame.draw.circle(self.image, (255, 100, 0), (10, 10), 10)
                elif projectile_type == 'water':
                    pygame.draw.circle(self.image, (0, 150, 255), (10, 10), 10)
                elif projectile_type == 'void':
                    pygame.draw.circle(self.image, (150, 0, 200), (10, 10), 10)
                elif projectile_type == 'ice':
                    pygame.draw.circle(
                        self.image, (150, 200, 255), (10, 10), 10)
                elif projectile_type == 'lightning':
                    pygame.draw.circle(
                        self.image, (255, 255, 100), (10, 10), 10)
                elif projectile_type == 'holy':
                    pygame.draw.circle(
                        self.image, (255, 255, 200), (10, 10), 10)
                else:
                    pygame.draw.circle(self.image, (255, 200, 0), (10, 10), 10)
        except Exception as e:
            print(f"Error loading projectile image: {e}")
            self.image = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(self.image, (255, 200, 0), (10, 10), 10)

        self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        self.rect.center = (self.x, self.y)

    def check_collision(self, collision_rects):
        for rect in collision_rects:
            if self.rect.colliderect(rect):
                return True
        return False

    def draw(self, surface, camera_x, camera_y):
        surface.blit(self.image, (self.x - camera_x -
                                  10, self.y - camera_y - 10))


class FloatingText:
    def __init__(self, x, y, text, color=(255, 0, 0)):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.timer = 60
        self.vel_y = -2
        self.alpha = 255

    def update(self):
        self.y += self.vel_y
        self.timer -= 1
        self.alpha = int((self.timer / 60) * 255)

    def draw(self, surface, camera_x, camera_y):
        if self.timer > 0:
            font = pygame.font.Font(None, 36)
            text_surf = font.render(self.text, True, self.color)
            text_surf.set_alpha(self.alpha)
            surface.blit(text_surf, (self.x - camera_x, self.y - camera_y))

    def is_alive(self):
        return self.timer > 0


class DialogueSystem:
    def __init__(self):
        self.active = False
        self.dialogues = []
        self.current_index = 0
        self.font = pygame.font.Font(None, 28)

    def start_dialogue(self, dialogues):
        self.dialogues = dialogues
        self.current_index = 0
        self.active = len(dialogues) > 0

    def next(self):
        if self.active:
            self.current_index += 1
            if self.current_index >= len(self.dialogues):
                self.active = False
                self.current_index = 0

    def draw(self, surface, screen_width, screen_height):
        if not self.active or not self.dialogues:
            return
        box_height = 120
        box_y = screen_height - box_height - 10
        box_rect = pygame.Rect(10, box_y, screen_width - 20, box_height)
        bg_surf = pygame.Surface((box_rect.width, box_rect.height))
        bg_surf.set_alpha(200)
        bg_surf.fill((20, 20, 40))
        surface.blit(bg_surf, box_rect.topleft)
        pygame.draw.rect(surface, (255, 255, 255), box_rect, 3)
        if self.current_index < len(self.dialogues):
            text = self.dialogues[self.current_index]
            words = text.split(' ')
            lines = []
            current_line = ""
            max_width = box_rect.width - 40
            for word in words:
                test_line = current_line + word + " "
                if self.font.size(test_line)[0] < max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)
            y_offset = box_y + 20
            for line in lines[:3]:
                text_surf = self.font.render(
                    line.strip(), True, (255, 255, 255))
                surface.blit(text_surf, (box_rect.x + 20, y_offset))
                y_offset += 30
        prompt = self.font.render(
            "Press SPACE to continue...", True, (200, 200, 200))
        surface.blit(prompt, (box_rect.x + 20, box_rect.bottom - 35))


class NPC:
    def __init__(self, x, y, tile_w, tile_h, npc_name='barman', dialogues=None):
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.size_multiplier = 2.0
        self.render_w = int(tile_w * self.size_multiplier)
        self.render_h = int(tile_h * self.size_multiplier)
        self.pixel_x = x
        self.pixel_y = y
        self.npc_name = npc_name
        self.interaction_range = 80
        if dialogues is None:
            self.dialogues = self.get_default_dialogues()
        else:
            self.dialogues = dialogues
        self.load_image()

    def get_default_dialogues(self):
        if self.npc_name.lower() == 'barman':
            return [
                "Welcome to the inn, weary traveler!",
                "I've been running this establishment for many years.",
                "If you're looking for adventure, head north to the main map.",
                "But beware! Dangerous creatures lurk in those lands.",
                "The boss rooms are especially perilous. Make sure you're prepared!",
                "Come back anytime you need rest and healing. Safe travels!"
            ]
        else:
            return [
                f"Greetings, traveler! I am {self.npc_name}.",
                "The world is full of dangers and mysteries.",
                "Be careful on your journey!"
            ]

    def load_image(self):
        try:
            img_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'image', f'{self.npc_name}.png')
            if os.path.exists(img_path):
                self.image = pygame.image.load(img_path).convert_alpha()
                self.image = pygame.transform.scale(
                    self.image, (self.render_w, self.render_h))
            else:
                self.image = pygame.Surface(
                    (self.render_w, self.render_h), pygame.SRCALPHA)
                if 'barman' in self.npc_name.lower():
                    pygame.draw.rect(self.image, (139, 69, 19),
                                     (10, 10, self.render_w-20, self.render_h-20))
                    pygame.draw.circle(
                        self.image, (255, 220, 177), (self.render_w//2, self.render_h//3), 15)
                else:
                    pygame.draw.rect(self.image, (100, 100, 200),
                                     (10, 10, self.render_w-20, self.render_h-20))
                    pygame.draw.circle(
                        self.image, (255, 220, 177), (self.render_w//2, self.render_h//3), 15)
        except Exception as e:
            print(f"Error loading NPC image: {e}")
            self.image = pygame.Surface(
                (self.render_w, self.render_h), pygame.SRCALPHA)
            pygame.draw.rect(self.image, (100, 100, 200),
                             (10, 10, self.render_w-20, self.render_h-20))

    def can_interact(self, player):
        distance = math.sqrt((self.pixel_x - player.pixel_x)
                             ** 2 + (self.pixel_y - player.pixel_y)**2)
        return distance <= self.interaction_range

    def draw(self, surface, camera_x, camera_y):
        surface.blit(self.image, (self.pixel_x -
                     camera_x, self.pixel_y - camera_y))
        font = pygame.font.Font(None, 20)
        name_surf = font.render(self.npc_name.upper(), True, (255, 255, 255))
        name_x = self.pixel_x - camera_x + \
            (self.render_w - name_surf.get_width()) // 2
        name_y = self.pixel_y - camera_y - 15
        bg_rect = pygame.Rect(
            name_x - 5, name_y - 2, name_surf.get_width() + 10, name_surf.get_height() + 4)
        bg_surf = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surf.set_alpha(180)
        bg_surf.fill((0, 0, 0))
        surface.blit(bg_surf, bg_rect.topleft)
        surface.blit(name_surf, (name_x, name_y))


class Player:
    def __init__(self, x, y, tile_w, tile_h):
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.size_multiplier = 1.0
        self.render_w = int(tile_w * self.size_multiplier)
        self.render_h = int(tile_h * self.size_multiplier)
        self.pixel_x = x * tile_w
        self.pixel_y = y * tile_h
        self.speed = 1
        self.run_speed = 3

        self.level = 1
        self.xp = 0
        self.xp_to_next_level = 100
        self.total_xp = 0

        self.base_max_health = 100
        self.base_max_stamina = 100
        self.base_attack_damage = 20
        self.base_crit_chance = 0.25

        self.max_health = self.base_max_health
        self.health = self.max_health
        self.max_stamina = self.base_max_stamina
        self.stamina = self.max_stamina
        self.attack_damage = self.base_attack_damage
        self.crit_chance = self.base_crit_chance

        self.stamina_regen = 0.3
        self.attack_cost = 25
        self.attack_range = 80
        self.attack_cooldown = 0
        self.crit_multiplier = 2.0
        self.state = State.IDLE
        self.hit_flash = 0

        self.animations = {'idle': [], 'walking': [],
                           'attacking': [], 'dying': []}
        self.current_direction = 'down'
        self.frame_index = 0
        self.animation_counter = 0.0
        self.animation_speed = 0.15
        self.attack_anim_timer = 0
        self.load_animations()

    def gain_xp(self, amount, game=None):
        self.xp += amount
        self.total_xp += amount
        if game:
            game.floating_texts.append(FloatingText(
                self.pixel_x + self.tile_w // 2,
                self.pixel_y - 20,
                f"+{amount} XP",
                (255, 255, 0)
            ))
        while self.xp >= self.xp_to_next_level:
            self.level_up(game)

    def level_up(self, game=None):
        self.xp -= self.xp_to_next_level
        self.level += 1
        self.xp_to_next_level = int(100 * (1.5 ** (self.level - 1)))
        old_max_health = self.max_health
        old_max_stamina = self.max_stamina
        self.max_health = self.base_max_health + (self.level - 1) * 15
        self.max_stamina = self.base_max_stamina + (self.level - 1) * 10
        self.attack_damage = self.base_attack_damage + (self.level - 1) * 5
        self.crit_chance = min(
            0.75, self.base_crit_chance + (self.level - 1) * 0.02)
        health_gained = self.max_health - old_max_health
        stamina_gained = self.max_stamina - old_max_stamina
        self.health = self.max_health
        self.stamina = self.max_stamina
        if game:
            game.message = f"LEVEL UP! Now Level {self.level}"
            game.message_timer = 120
            game.floating_texts.append(FloatingText(
                self.pixel_x + self.tile_w // 2,
                self.pixel_y - 40,
                f"LEVEL {self.level}!",
                (0, 255, 255)
            ))
            game.play_sound('level_up')
        print(f"Level Up! Now Level {self.level}")
        print(f"  Max Health: {self.max_health} (+{health_gained})")
        print(f"  Max Stamina: {self.max_stamina} (+{stamina_gained})")
        print(f"  Attack Damage: {self.attack_damage}")
        print(f"  Crit Chance: {int(self.crit_chance * 100)}%")

    def load_animations(self):
        base = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'image')
        sizes = (max(1, self.render_w), max(1, self.render_h))

        def load_folder(name):
            path = os.path.join(base, name)
            frames = []
            try:
                if os.path.isdir(path):
                    files = sorted([f for f in os.listdir(
                        path) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
                    for fn in files:
                        try:
                            img = pygame.image.load(
                                os.path.join(path, fn)).convert_alpha()
                            img = pygame.transform.scale(img, sizes)
                            frames.append(img)
                        except Exception:
                            pass
            except Exception:
                pass
            if not frames:
                placeholder = pygame.Surface(sizes, pygame.SRCALPHA)
                pygame.draw.rect(placeholder, (100, 150, 255),
                                 (0, 0, sizes[0], sizes[1]))
                frames = [placeholder]
            return frames

        self.animations['idle'] = load_folder('idle')
        self.animations['walking'] = load_folder('walking')
        self.animations['attacking'] = load_folder('attacking')
        self.animations['dying'] = load_folder('dying')

    def set_tile_size(self, tile_w, tile_h):
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.render_w = int(tile_w * self.size_multiplier)
        self.render_h = int(tile_h * self.size_multiplier)
        self.load_animations()

    def shoot_projectile(self, target_x, target_y):
        if self.stamina >= self.attack_cost and self.attack_cooldown == 0 and self.state != State.DEAD:
            self.stamina -= self.attack_cost
            self.attack_cooldown = 30
            is_crit = random.random() < self.crit_chance
            damage = self.attack_damage * \
                self.crit_multiplier if is_crit else self.attack_damage
            center_x = self.pixel_x + self.tile_w // 2
            center_y = self.pixel_y + self.tile_h // 2
            return Projectile(center_x, center_y, target_x, target_y, damage, is_enemy=False), is_crit
        return None, False

    def attack(self, enemies):
        if self.stamina >= self.attack_cost and self.attack_cooldown == 0 and self.state != State.DEAD:
            self.stamina -= self.attack_cost
            self.attack_cooldown = 30
            self.state = State.ATTACKING
            self.attack_anim_timer = len(
                self.animations.get('attacking', [])) * 6
            hit_any = False
            for enemy in enemies:
                if enemy.state != State.DEAD:
                    distance = ((self.pixel_x - enemy.pixel_x) **
                                2 + (self.pixel_y - enemy.pixel_y)**2)**0.5
                    if distance <= self.attack_range:
                        is_crit = random.random() < self.crit_chance
                        damage = self.attack_damage * \
                            self.crit_multiplier if is_crit else self.attack_damage
                        enemy.take_damage(damage, is_crit)
                        hit_any = True
            return hit_any
        return False

    def take_damage(self, damage):
        if self.state != State.DEAD:
            self.health -= damage
            self.hit_flash = 10
            if self.health <= 0:
                self.health = 0
                self.state = State.DEAD
                return True
            else:
                self.state = State.HURT
            return False

    def update_combat(self):
        if self.state != State.ATTACKING:
            self.stamina = min(
                self.max_stamina, self.stamina + self.stamina_regen)
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.hit_flash > 0:
            self.hit_flash -= 1
        if self.state == State.HURT and self.hit_flash == 0:
            self.state = State.IDLE
        if getattr(self, 'attack_anim_timer', 0) > 0:
            self.attack_anim_timer -= 1
            if self.attack_anim_timer <= 0:
                if self.state == State.ATTACKING:
                    self.state = State.IDLE

    def handle_input(self, keys, collision_rects, map_width, map_height):
        moving = False
        if self.state != State.DEAD:
            current_speed = self.run_speed if (
                keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else self.speed
            dx = dy = 0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                dx -= current_speed
                self.current_direction = 'left'
                moving = True
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                dx += current_speed
                self.current_direction = 'right'
                moving = True
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                dy -= current_speed
                if not (keys[pygame.K_LEFT] or keys[pygame.K_a] or keys[pygame.K_RIGHT] or keys[pygame.K_d]):
                    self.current_direction = 'up'
                moving = True
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                dy += current_speed
                if not (keys[pygame.K_LEFT] or keys[pygame.K_a] or keys[pygame.K_RIGHT] or keys[pygame.K_d]):
                    self.current_direction = 'down'
                moving = True
            if dx != 0 and dy != 0:
                dx *= 0.707
                dy *= 0.707
            new_rect = pygame.Rect(
                self.pixel_x+dx, self.pixel_y+dy, self.tile_w, self.tile_h)
            if not any(new_rect.colliderect(r) for r in collision_rects):
                if 0 <= new_rect.x <= map_width*self.tile_w - self.tile_w and 0 <= new_rect.y <= map_height*self.tile_h - self.tile_h:
                    self.pixel_x += dx
                    self.pixel_y += dy

        if self.state == State.DEAD:
            anim_key = 'dying'
        elif self.state == State.ATTACKING:
            anim_key = 'attacking'
        elif moving:
            anim_key = 'walking'
        else:
            anim_key = 'idle'

        if not hasattr(self, 'current_anim_key') or self.current_anim_key != anim_key:
            self.current_anim_key = anim_key
            self.frame_index = 0
            self.animation_counter = 0.0

        frames = self.animations.get(self.current_anim_key, [])
        if frames:
            if self.state == State.DEAD:
                if self.frame_index < len(frames) - 1:
                    self.animation_counter += self.animation_speed
                    if self.animation_counter >= 1.0:
                        self.animation_counter = 0.0
                        self.frame_index += 1
            else:
                self.animation_counter += self.animation_speed
                if self.animation_counter >= 1.0:
                    self.animation_counter = 0.0
                    self.frame_index = (self.frame_index + 1) % len(frames)
        else:
            self.frame_index = 0
            self.animation_counter = 0.0

    def draw(self, surface, camera_x, camera_y):
        anim_key = getattr(self, 'current_anim_key', 'idle')
        frames = self.animations.get(anim_key, [])
        if not frames:
            img = pygame.Surface(
                (self.render_w, self.render_h), pygame.SRCALPHA)
            pygame.draw.rect(img, (100, 150, 255),
                             (0, 0, self.render_w, self.render_h))
        else:
            idx = max(0, min(self.frame_index, len(frames)-1))
            img = frames[idx]
        if self.current_direction == 'left':
            img = pygame.transform.flip(img, True, False)
        if self.hit_flash > 0:
            flash_img = img.copy()
            flash_img.fill((255, 255, 255, 100),
                           special_flags=pygame.BLEND_RGB_ADD)
            surface.blit(flash_img, (self.pixel_x -
                         camera_x, self.pixel_y - camera_y))
        else:
            surface.blit(img, (self.pixel_x - camera_x,
                         self.pixel_y - camera_y))


class Slime:
    def __init__(self, x, y, tile_w, tile_h, slime_type='red_slime'):
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.size_multiplier = 2.0
        self.render_w = int(tile_w * self.size_multiplier)
        self.render_h = int(tile_h * self.size_multiplier)
        self.pixel_x = x
        self.pixel_y = y
        self.slime_type = slime_type
        if slime_type == 'blue_slime':
            self.max_health = 50
            self.speed = 1.5
            self.attack_damage = 5
            self.attack_range = 60
        elif slime_type == 'yellow_slime':
            self.max_health = 60
            self.speed = 1.0
            self.attack_damage = 7
            self.attack_range = 70
        else:
            self.max_health = 55
            self.speed = 1.3
            self.attack_damage = 6
            self.attack_range = 65
        self.health = self.max_health
        self.attack_cooldown = 0
        self.state = State.IDLE
        self.hit_flash = 0
        self.detection_range = 200
        self.wander_timer = 0
        self.wander_direction = [0, 0]
        self.frame_index = 0
        self.animation_counter = 0.0
        self.animation_speed = 0.15
        self.load_animations()

    def load_animations(self):
        base = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'image')
        self.idle_frames = []
        self.attack_frames = []
        for i in range(2):
            try:
                if i == 0:
                    path = os.path.join(base, f'{self.slime_type}_idle.png')
                else:
                    path = os.path.join(base, f'{self.slime_type}_idle{i}.png')
                if os.path.exists(path):
                    img = pygame.image.load(path).convert_alpha()
                    img = pygame.transform.scale(
                        img, (self.render_w, self.render_h))
                    self.idle_frames.append(img)
            except Exception as e:
                print(f"Could not load {path}: {e}")
        for i in range(7):
            try:
                if i == 0:
                    path = os.path.join(base, f'{self.slime_type}_attack.png')
                else:
                    path = os.path.join(
                        base, f'{self.slime_type}_attack{i}.png')
                if os.path.exists(path):
                    img = pygame.image.load(path).convert_alpha()
                    img = pygame.transform.scale(
                        img, (self.render_w, self.render_h))
                    self.attack_frames.append(img)
            except Exception as e:
                print(f"Could not load {path}: {e}")
        if not self.idle_frames:
            placeholder = pygame.Surface(
                (self.render_w, self.render_h), pygame.SRCALPHA)
            if 'blue' in self.slime_type:
                pygame.draw.circle(placeholder, (100, 100, 255),
                                   (self.render_w//2, self.render_h//2), self.render_w//3)
            elif 'yellow' in self.slime_type:
                pygame.draw.circle(placeholder, (255, 255, 100),
                                   (self.render_w//2, self.render_h//2), self.render_w//3)
            else:
                pygame.draw.circle(placeholder, (255, 100, 100),
                                   (self.render_w//2, self.render_h//2), self.render_w//3)
            self.idle_frames = [placeholder]
            self.attack_frames = [placeholder]

    def take_damage(self, damage, is_crit=False):
        if self.state != State.DEAD:
            self.health -= damage
            self.hit_flash = 10
            self.is_crit = is_crit
            if self.health <= 0:
                self.health = 0
                self.state = State.DEAD
                xp_reward = 20 + (self.max_health // 10)
                return True, xp_reward
            else:
                self.state = State.HURT
            return False, 0

    def update(self, player, collision_rects, map_width, map_height, game=None):
        if self.state == State.DEAD:
            return
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.hit_flash > 0:
            self.hit_flash -= 1
        if self.state == State.HURT and self.hit_flash == 0:
            self.state = State.IDLE
        distance = ((self.pixel_x - player.pixel_x)**2 +
                    (self.pixel_y - player.pixel_y)**2)**0.5
        if distance <= self.detection_range and player.state != State.DEAD:
            dx = player.pixel_x - self.pixel_x
            dy = player.pixel_y - self.pixel_y
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                dx = (dx / length) * self.speed
                dy = (dy / length) * self.speed
                new_rect = pygame.Rect(
                    self.pixel_x + dx, self.pixel_y + dy, self.tile_w, self.tile_h)
                if not any(new_rect.colliderect(r) for r in collision_rects):
                    self.pixel_x += dx
                    self.pixel_y += dy
            if distance <= self.attack_range and self.attack_cooldown == 0:
                player.take_damage(self.attack_damage)
                if game:
                    game.play_sound('taking_damage')
                self.attack_cooldown = 60
                self.state = State.ATTACKING
        else:
            if self.wander_timer <= 0:
                self.wander_timer = random.randint(60, 180)
                self.wander_direction = [
                    random.uniform(-1, 1), random.uniform(-1, 1)]
            else:
                self.wander_timer -= 1
                dx = self.wander_direction[0] * self.speed * 0.5
                dy = self.wander_direction[1] * self.speed * 0.5
                new_rect = pygame.Rect(
                    self.pixel_x + dx, self.pixel_y + dy, self.tile_w, self.tile_h)
                if not any(new_rect.colliderect(r) for r in collision_rects):
                    if 0 <= new_rect.x <= map_width*self.tile_w and 0 <= new_rect.y <= map_height*self.tile_h:
                        self.pixel_x += dx
                        self.pixel_y += dy
        frames = self.attack_frames if self.state == State.ATTACKING else self.idle_frames
        if frames:
            self.animation_counter += self.animation_speed
            if self.animation_counter >= 1.0:
                self.animation_counter = 0.0
                if self.state == State.ATTACKING:
                    self.frame_index += 1
                    if self.frame_index >= len(self.attack_frames):
                        self.frame_index = 0
                        self.state = State.IDLE
                else:
                    self.frame_index = (self.frame_index + 1) % len(frames)

    def draw(self, surface, camera_x, camera_y):
        frames = self.attack_frames if self.state == State.ATTACKING else self.idle_frames
        if frames:
            idx = max(0, min(self.frame_index, len(frames)-1))
            img = frames[idx].copy()
        else:
            img = pygame.Surface(
                (self.render_w, self.render_h), pygame.SRCALPHA)
        if self.hit_flash > 0:
            img.fill((255, 255, 255, 100), special_flags=pygame.BLEND_RGB_ADD)
        if self.state == State.DEAD:
            img.set_alpha(100)
        surface.blit(img, (self.pixel_x - camera_x, self.pixel_y - camera_y))
        if self.state != State.DEAD:
            bar_width = self.render_w
            bar_height = 5
            bar_x = self.pixel_x - camera_x
            bar_y = self.pixel_y - camera_y - 10
            pygame.draw.rect(surface, (100, 0, 0),
                             (bar_x, bar_y, bar_width, bar_height))
            health_width = int((self.health / self.max_health) * bar_width)
            pygame.draw.rect(surface, (0, 255, 0),
                             (bar_x, bar_y, health_width, bar_height))


class Tower:
    def __init__(self, x, y, tile_w, tile_h, tower_type='fire'):
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.size_multiplier = 3.0
        self.render_w = int(tile_w * self.size_multiplier)
        self.render_h = int(tile_h * self.size_multiplier)
        self.pixel_x = x
        self.pixel_y = y
        self.tower_type = tower_type
        if tower_type == 'fire':
            self.max_health = 100
            self.attack_damage = 12
            self.shoot_interval = 90
            self.detection_range = 400
        elif tower_type == 'water':
            self.max_health = 120
            self.attack_damage = 10
            self.shoot_interval = 100
            self.detection_range = 450
        elif tower_type == 'void':
            self.max_health = 80
            self.attack_damage = 15
            self.shoot_interval = 80
            self.detection_range = 380
        elif tower_type == 'ice':
            self.max_health = 110
            self.attack_damage = 11
            self.shoot_interval = 95
            self.detection_range = 420
        elif tower_type == 'lightning':
            self.max_health = 90
            self.attack_damage = 14
            self.shoot_interval = 75
            self.detection_range = 400
        elif tower_type == 'holy':
            self.max_health = 130
            self.attack_damage = 13
            self.shoot_interval = 110
            self.detection_range = 500
        else:
            self.max_health = 100
            self.attack_damage = 10
            self.shoot_interval = 100
            self.detection_range = 400
        self.health = self.max_health
        self.state = State.IDLE
        self.hit_flash = 0
        self.shoot_cooldown = 0
        self.load_image()

    def load_image(self):
        try:
            img_path = os.path.join(os.path.dirname(os.path.abspath(
                __file__)), 'image', f'tower_{self.tower_type}.png')
            if os.path.exists(img_path):
                self.image = pygame.image.load(img_path).convert_alpha()
                self.image = pygame.transform.scale(
                    self.image, (self.render_w, self.render_h))
                print(f"Loaded tower image: tower_{self.tower_type}.png")
            else:
                self.image = pygame.Surface(
                    (self.render_w, self.render_h), pygame.SRCALPHA)
                if self.tower_type == 'fire':
                    color = (255, 100, 0)
                elif self.tower_type == 'water':
                    color = (0, 100, 255)
                elif self.tower_type == 'void':
                    color = (100, 0, 150)
                elif self.tower_type == 'ice':
                    color = (150, 200, 255)
                elif self.tower_type == 'lightning':
                    color = (255, 255, 100)
                elif self.tower_type == 'holy':
                    color = (255, 255, 200)
                else:
                    color = (150, 150, 150)
                pygame.draw.rect(
                    self.image, color, (10, self.render_h//2, self.render_w-20, self.render_h//2-10))
                pygame.draw.circle(self.image, (200, 200, 200),
                                   (self.render_w//2, self.render_h//3), 15)
                pygame.draw.rect(self.image, (80, 80, 80),
                                 (self.render_w//2-5, self.render_h//3-20, 10, 20))
                print(f"Created fallback tower graphics for {self.tower_type}")
        except Exception as e:
            print(f"Error loading tower image: {e}")
            self.image = pygame.Surface(
                (self.render_w, self.render_h), pygame.SRCALPHA)
            pygame.draw.rect(self.image, (150, 150, 150),
                             (10, 10, self.render_w-20, self.render_h-20))

    def take_damage(self, damage, is_crit=False):
        if self.state != State.DEAD:
            self.health -= damage
            self.hit_flash = 10
            self.is_crit = is_crit
            if self.health <= 0:
                self.health = 0
                self.state = State.DEAD
                xp_reward = 100
                return True, xp_reward
            else:
                self.state = State.HURT
            return False, 0

    def update(self, player):
        if self.state == State.DEAD:
            return None
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.hit_flash > 0:
            self.hit_flash -= 1
        if self.state == State.HURT and self.hit_flash == 0:
            self.state = State.IDLE
        distance = ((self.pixel_x - player.pixel_x)**2 +
                    (self.pixel_y - player.pixel_y)**2)**0.5
        if distance <= self.detection_range and self.shoot_cooldown == 0 and player.state != State.DEAD:
            self.shoot_cooldown = self.shoot_interval
            center_x = self.pixel_x + self.render_w // 2
            center_y = self.pixel_y + self.render_h // 4
            target_x = player.pixel_x + player.tile_w // 2
            target_y = player.pixel_y + player.tile_h // 2
            return Projectile(center_x, center_y, target_x, target_y, self.attack_damage,
                              is_enemy=True, projectile_type=self.tower_type)
        return None

    def draw(self, surface, camera_x, camera_y):
        img = self.image.copy()
        if self.hit_flash > 0:
            img.fill((255, 255, 255, 100), special_flags=pygame.BLEND_RGB_ADD)
        if self.state == State.DEAD:
            img.set_alpha(100)
        surface.blit(img, (self.pixel_x - camera_x, self.pixel_y - camera_y))
        if self.state != State.DEAD:
            bar_width = self.render_w
            bar_height = 6
            bar_x = self.pixel_x - camera_x
            bar_y = self.pixel_y - camera_y - 12
            pygame.draw.rect(surface, (100, 0, 0),
                             (bar_x, bar_y, bar_width, bar_height))
            health_width = int((self.health / self.max_health) * bar_width)
            pygame.draw.rect(surface, (255, 0, 0),
                             (bar_x, bar_y, health_width, bar_height))


class Boss:
    def __init__(self, x, y, tile_w, tile_h):
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.size_multiplier = 4.0
        self.render_w = int(tile_w * self.size_multiplier)
        self.render_h = int(tile_h * self.size_multiplier)
        self.pixel_x = x
        self.pixel_y = y
        self.max_health = 200
        self.health = 200
        self.attack_damage = 15
        self.state = State.IDLE
        self.hit_flash = 0
        self.shoot_cooldown = 0
        self.shoot_interval = 120
        self.detection_range = 400
        self.load_image()

    def load_image(self):
        try:
            self.image = pygame.Surface(
                (self.render_w, self.render_h), pygame.SRCALPHA)
            pygame.draw.rect(self.image, (150, 0, 150),
                             (0, 0, self.render_w, self.render_h))
            pygame.draw.circle(self.image, (200, 0, 200),
                               (self.render_w//2, self.render_h//2), 40)
        except:
            self.image = pygame.Surface(
                (self.render_w, self.render_h), pygame.SRCALPHA)
            pygame.draw.rect(self.image, (150, 0, 150),
                             (0, 0, self.render_w, self.render_h))

    def take_damage(self, damage, is_crit=False):
        if self.state != State.DEAD:
            self.health -= damage
            self.hit_flash = 10
            self.is_crit = is_crit
            if self.health <= 0:
                self.health = 0
                self.state = State.DEAD
                xp_reward = 50
                return True, xp_reward
            else:
                self.state = State.HURT
            return False, 0

    def update(self, player):
        if self.state == State.DEAD:
            return None
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.hit_flash > 0:
            self.hit_flash -= 1
        if self.state == State.HURT and self.hit_flash == 0:
            self.state = State.IDLE
        distance = ((self.pixel_x - player.pixel_x)**2 +
                    (self.pixel_y - player.pixel_y)**2)**0.5
        if distance <= self.detection_range and self.shoot_cooldown == 0 and player.state != State.DEAD:
            self.shoot_cooldown = self.shoot_interval
            center_x = self.pixel_x + self.render_w // 2
            center_y = self.pixel_y + self.render_h // 2
            target_x = player.pixel_x + player.tile_w // 2
            target_y = player.pixel_y + player.tile_h // 2
            return Projectile(center_x, center_y, target_x, target_y, self.attack_damage,
                              is_enemy=True, projectile_type='void')
        return None

    def draw(self, surface, camera_x, camera_y):
        img = self.image.copy()
        if self.hit_flash > 0:
            img.fill((255, 255, 255, 100), special_flags=pygame.BLEND_RGB_ADD)
        if self.state == State.DEAD:
            img.set_alpha(100)
        surface.blit(img, (self.pixel_x - camera_x, self.pixel_y - camera_y))
        if self.state != State.DEAD:
            bar_width = self.render_w
            bar_height = 8
            bar_x = self.pixel_x - camera_x
            bar_y = self.pixel_y - camera_y - 15
            pygame.draw.rect(surface, (100, 0, 0),
                             (bar_x, bar_y, bar_width, bar_height))
            health_width = int((self.health / self.max_health) * bar_width)
            pygame.draw.rect(surface, (255, 0, 0),
                             (bar_x, bar_y, health_width, bar_height))


class Camera:
    def __init__(self, screen_width, screen_height, map_width, map_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.map_width = map_width
        self.map_height = map_height
        self.x = 0
        self.y = 0

    def update(self, target_x, target_y, target_width, target_height):
        self.x = target_x + target_width // 2 - self.screen_width // 2
        self.y = target_y + target_height // 2 - self.screen_height // 2
        self.x = max(0, min(self.x, self.map_width - self.screen_width))
        self.y = max(0, min(self.y, self.map_height - self.screen_height))

    def update_screen_size(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height


class GameMap:
    def __init__(self, tmx_file):
        self.current_map_file = tmx_file
        try:
            self.tmx_data = load_pygame(tmx_file)
        except Exception as e:
            print(f"Error loading TMX file: {e}")
            print("Attempting to auto-fix tileset source paths...")
            import xml.etree.ElementTree as ET
            tmx_dir = os.path.dirname(os.path.abspath(tmx_file))
            tree = ET.parse(tmx_file)
            root = tree.getroot()
            changed = False
            for tileset in root.findall('tileset'):
                src = tileset.get('source')
                if src:
                    base = os.path.basename(src)
                    candidate = os.path.join(tmx_dir, base)
                    if os.path.exists(candidate):
                        tileset.set('source', candidate)
                        changed = True
            if changed:
                fixed_path = os.path.join(tmx_dir, os.path.basename(
                    tmx_file).replace('.tmx', '_fixed.tmx'))
                tree.write(fixed_path, encoding='utf-8', xml_declaration=True)
                self.tmx_data = load_pygame(fixed_path)
            else:
                raise
        self.tile_w = self.tmx_data.tilewidth
        self.tile_h = self.tmx_data.tileheight
        self.width = self.tmx_data.width
        self.height = self.tmx_data.height
        self.collision_rects = self.build_collision_rects()
        self.teleports = self.build_teleports()
        self.bosses = self.build_bosses()
        self.towers = self.build_towers()
        self.npcs = self.build_npcs()

    def build_collision_rects(self):
        rects = []
        layers = list(self.tmx_data.visible_layers)
        if layers:
            bottom_layer = layers[0]
            if isinstance(bottom_layer, pytmx.TiledTileLayer):
                if bottom_layer.properties.get("blocked") or bottom_layer.name.lower() == "collision":
                    for x, y, gid in bottom_layer.tiles():
                        if gid != 0:
                            rects.append(pygame.Rect(x * self.tmx_data.tilewidth, y * self.tmx_data.tileheight,
                                                     self.tmx_data.tilewidth, self.tmx_data.tileheight))
                    return rects
        for layer in self.tmx_data.visible_layers:
            if isinstance(layer, pytmx.TiledTileLayer):
                if layer.properties.get("blocked"):
                    for x, y, gid in layer.tiles():
                        if gid != 0:
                            rects.append(pygame.Rect(x * self.tmx_data.tilewidth, y * self.tmx_data.tileheight,
                                                     self.tmx_data.tilewidth, self.tmx_data.tileheight))
        return rects

    def build_teleports(self):
        teleports = []
        try:
            for obj in getattr(self.tmx_data, 'objects', []):
                obj_type = getattr(obj, 'type', '') or getattr(obj, 'name', '')
                if str(obj_type).lower() == 'teleport':
                    props = getattr(obj, 'properties', {}) or {}
                    dest = props.get('dest') or props.get(
                        'map') or props.get('destination')
                    dest_x = props.get('dest_x')
                    dest_y = props.get('dest_y')
                    rect = pygame.Rect(int(obj.x), int(obj.y), int(getattr(obj, 'width', 0) or 1),
                                       int(getattr(obj, 'height', 0) or 1))
                    teleports.append(
                        {'rect': rect, 'dest': dest, 'dest_x': dest_x, 'dest_y': dest_y, 'obj': obj})
        except Exception:
            pass
        return teleports

    def build_bosses(self):
        bosses = []
        try:
            for obj in getattr(self.tmx_data, 'objects', []):
                obj_type = getattr(obj, 'type', '') or getattr(obj, 'name', '')
                if str(obj_type).lower() == 'boss':
                    boss = Boss(int(obj.x), int(obj.y),
                                self.tile_w, self.tile_h)
                    bosses.append(boss)
                    print(f"Found boss at ({obj.x}, {obj.y})")
        except Exception as e:
            print(f"Error loading bosses: {e}")
        return bosses

    def build_towers(self):
        towers = []
        image_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'image')
        available_towers = []
        tower_types = ['fire', 'water', 'void', 'ice', 'lightning', 'holy']
        for tower_type in tower_types:
            tower_img_path = os.path.join(image_dir, f'tower_{tower_type}.png')
            if os.path.exists(tower_img_path):
                available_towers.append(tower_type)
                print(f"Found tower image: tower_{tower_type}.png")
        try:
            map_name = os.path.basename(
                getattr(self, 'current_map_file', '')).lower()
            if 'winter' in map_name or 'boss' in map_name:
                tower_configs = [
                    {'x': 5, 'y': 5, 'type': 'ice'},
                    {'x': self.width - 8, 'y': 5, 'type': 'ice'},
                    {'x': 5, 'y': self.height - 8, 'type': 'water'},
                    {'x': self.width - 8, 'y': self.height - 8, 'type': 'water'},
                ]
            elif 'angel' in map_name:
                tower_configs = [
                    {'x': self.width // 2 - 3, 'y': 5, 'type': 'holy'},
                    {'x': 5, 'y': self.height // 2, 'type': 'holy'},
                    {'x': self.width - 8, 'y': self.height // 2, 'type': 'holy'},
                ]
            elif 'fire' in map_name or 'lava' in map_name:
                tower_configs = [
                    {'x': 7, 'y': 7, 'type': 'fire'},
                    {'x': self.width - 10, 'y': 7, 'type': 'fire'},
                    {'x': self.width // 2, 'y': self.height - 10, 'type': 'void'},
                ]
            else:
                tower_configs = [
                    {'x': 10, 'y': 10, 'type': 'fire'},
                    {'x': self.width - 13, 'y': 10, 'type': 'water'},
                ]
            for config in tower_configs:
                tower_type = config['type']
                if tower_type in available_towers or True:
                    x_pixel = config['x'] * self.tile_w
                    y_pixel = config['y'] * self.tile_h
                    test_rect = pygame.Rect(
                        x_pixel, y_pixel, self.tile_w * 3, self.tile_h * 3)
                    if not any(test_rect.colliderect(r) for r in self.collision_rects):
                        tower = Tower(x_pixel, y_pixel, self.tile_w,
                                      self.tile_h, tower_type)
                        towers.append(tower)
                        print(
                            f"Spawned {tower_type} tower at ({x_pixel}, {y_pixel})")
        except Exception as e:
            print(f"Error building towers: {e}")
        return towers

    def build_npcs(self):
        npcs = []
        try:
            for obj in getattr(self.tmx_data, 'objects', []):
                obj_type = getattr(obj, 'type', '') or getattr(obj, 'name', '')
                obj_type_lower = str(obj_type).lower()
                if 'npc' in obj_type_lower or 'barman' in obj_type_lower or 'merchant' in obj_type_lower:
                    props = getattr(obj, 'properties', {}) or {}
                    custom_dialogues = []
                    i = 1
                    while True:
                        dialogue_key = f'dialogue{i}'
                        if dialogue_key in props:
                            custom_dialogues.append(props[dialogue_key])
                            i += 1
                        else:
                            break
                    npc_name = 'barman'
                    if 'barman' in obj_type_lower:
                        npc_name = 'barman'
                    elif 'merchant' in obj_type_lower:
                        npc_name = 'merchant'
                    else:
                        npc_name = obj_type
                    npc = NPC(int(obj.x), int(obj.y), self.tile_w, self.tile_h,
                              npc_name, custom_dialogues if custom_dialogues else None)
                    npcs.append(npc)
                    print(f"Found NPC '{npc_name}' at ({obj.x}, {obj.y})")
        except Exception as e:
            print(f"Error loading NPCs: {e}")
        return npcs

    def draw(self, surface, camera_x, camera_y):
        for layer in self.tmx_data.visible_layers:
            if isinstance(layer, pytmx.TiledTileLayer):
                for x, y, image in layer.tiles():
                    if image:
                        surface.blit(
                            image, (x*self.tile_w - camera_x, y*self.tile_h - camera_y))


def draw_ui_bar(surface, x, y, w, h, value, max_value, color, bg_color, label):
    font = pygame.font.Font(None, 20)
    label_surf = font.render(label, True, (255, 255, 255))
    surface.blit(label_surf, (x, y - 18))
    pygame.draw.rect(surface, bg_color, (x, y, w, h))
    fill_w = int((value / max_value) * w)
    pygame.draw.rect(surface, color, (x, y, fill_w, h))
    pygame.draw.rect(surface, (0, 0, 0), (x, y, w, h), 2)
    text = font.render(f"{int(value)}/{int(max_value)}", True, (255, 255, 255))
    text_rect = text.get_rect(center=(x + w//2, y + h//2))
    surface.blit(text, text_rect)


def spawn_slimes_randomly(map_obj, count=5):
    slimes = []
    collision_rects = map_obj.collision_rects
    for _ in range(count):
        slime_type = random.choice(['red_slime', 'blue_slime', 'yellow_slime'])
        max_attempts = 50
        for attempt in range(max_attempts):
            x = random.randint(5, map_obj.width - 5) * map_obj.tile_w
            y = random.randint(5, map_obj.height - 5) * map_obj.tile_h
            test_rect = pygame.Rect(
                x, y, map_obj.tile_w * 2, map_obj.tile_h * 2)
            if not any(test_rect.colliderect(r) for r in collision_rects):
                slime = Slime(x, y, map_obj.tile_w, map_obj.tile_h, slime_type)
                slimes.append(slime)
                break
    return slimes


class Game:
    def __init__(self, tmx_file, fullscreen=True):
        pygame.init()
        pygame.mixer.init()
        self.sounds = {}
        self.load_sounds()
        self.default_width = 800
        self.default_height = 600
        self.fullscreen = fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.screen_width = self.screen.get_width()
            self.screen_height = self.screen.get_height()
        else:
            self.screen_width = self.default_width
            self.screen_height = self.default_height
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height))
        pygame.display.set_caption(
            "Medieval RPG - Click to Shoot, SPACE to Attack/Continue, E to Interact")
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_map = GameMap(tmx_file)
        self.current_map = tmx_file
        self.debug_draw_teleports = False
        self.player = Player(166, 57, self.game_map.tile_w,
                             self.game_map.tile_h)
        self.teleport_cooldown = 0
        self.teleport_marker_rect = None
        self.teleport_marker_timer = 0
        self.teleport_marker_duration = 300
        self.slimes = spawn_slimes_randomly(self.game_map, count=8)
        self.bosses = self.game_map.bosses
        self.towers = self.game_map.towers
        self.npcs = self.game_map.npcs
        self.projectiles = []
        self.floating_texts = []
        self.camera = Camera(self.screen_width, self.screen_height,
                             self.game_map.width * self.game_map.tile_w,
                             self.game_map.height * self.game_map.tile_h)
        self.font = pygame.font.Font(None, 24)
        self.message = ""
        self.message_timer = 0
        self.dialogue = DialogueSystem()
        self.current_music = None
        self.load_music(tmx_file)
        self.nearby_npc = None
        self.start_intro_dialogue()

    def start_intro_dialogue(self):
        intro_dialogues = [
            "Welcome, brave warrior! Your journey begins here.",
            "The realm is in great danger. Dark forces threaten our land.",
            "You must defeat the monsters and face the powerful bosses.",
            "Press SPACE to attack nearby enemies, or click to shoot projectiles.",
            "Press E to interact with NPCs and teleport.",
            "Collect your strength and prepare for battle!",
            "Good luck, hero. The fate of the realm rests in your hands."
        ]
        self.dialogue.start_dialogue(intro_dialogues)

    def load_sounds(self):
        sound_names = ['projectile', 'attacking',
                       'dying', 'taking_damage', 'level_up']
        sound_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'sound')
        if not os.path.exists(sound_dir):
            try:
                os.makedirs(sound_dir)
                print(f"Created sound directory: {sound_dir}")
            except Exception as e:
                print(f"Could not create sound directory: {e}")
        for sound_name in sound_names:
            try:
                sound_path = os.path.join(sound_dir, f'{sound_name}.wav')
                if os.path.exists(sound_path):
                    self.sounds[sound_name] = pygame.mixer.Sound(sound_path)
                    self.sounds[sound_name].set_volume(0.5)
                    print(f"Loaded sound: {sound_name}.wav")
                else:
                    sound_path_ogg = os.path.join(
                        sound_dir, f'{sound_name}.ogg')
                    if os.path.exists(sound_path_ogg):
                        self.sounds[sound_name] = pygame.mixer.Sound(
                            sound_path_ogg)
                        self.sounds[sound_name].set_volume(0.5)
                        print(f"Loaded sound: {sound_name}.ogg")
                    else:
                        print(
                            f"Sound file not found: {sound_path} or {sound_path_ogg}")
                        self.sounds[sound_name] = None
            except Exception as e:
                print(f"Error loading sound {sound_name}: {e}")
                self.sounds[sound_name] = None

    def play_sound(self, sound_name):
        if sound_name in self.sounds and self.sounds[sound_name]:
            try:
                self.sounds[sound_name].play()
            except Exception as e:
                print(f"Error playing sound {sound_name}: {e}")

    def load_music(self, tmx_file):
        try:
            map_name = os.path.basename(tmx_file).replace('.tmx', '')
            music_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'music', f'{map_name}.mp3')
            if os.path.exists(music_path) and music_path != self.current_music:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play(-1)
                self.current_music = music_path
                print(f"Playing music: {music_path}")
            elif not os.path.exists(music_path):
                print(f"Music file not found: {music_path}")
                generic_music = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), 'music', 'background.mp3')
                if os.path.exists(generic_music) and generic_music != self.current_music:
                    pygame.mixer.music.load(generic_music)
                    pygame.mixer.music.set_volume(0.5)
                    pygame.mixer.music.play(-1)
                    self.current_music = generic_music
                    print(f"Playing generic music: {generic_music}")
        except Exception as e:
            print(f"Could not load music: {e}")

    def load_map(self, tmx_file, teleport_obj=None):
        try:
            new_map = GameMap(tmx_file)
        except Exception as e:
            self.message = f"Failed to load map: {os.path.basename(tmx_file)}"
            self.message_timer = 60
            print(f"load_map error: {e}")
            return
        self.game_map = new_map
        self.current_map = tmx_file
        try:
            self.player.set_tile_size(
                self.game_map.tile_w, self.game_map.tile_h)
        except Exception:
            pass
        if teleport_obj:
            teleports = getattr(self.game_map, 'teleports', [])
            src_rect = teleport_obj.get('rect')
            next_tp = None
            try:
                src_map = os.path.basename(
                    self.current_map).lower() if self.current_map else ''
            except Exception:
                src_map = ''
            dest_map_name = os.path.basename(tmx_file).lower()
            if src_map == 'home_inn_1.tmx' and dest_map_name == 'main_map.tmx' and len(teleports) >= 2:
                next_tp = teleports[1]
            else:
                if teleports and src_rect is not None:
                    found = None
                    for i, tp in enumerate(teleports):
                        r = tp.get('rect')
                        if r and r.x == src_rect.x and r.y == src_rect.y and r.width == src_rect.width and r.height == src_rect.height:
                            found = i
                            break
                    if found is not None:
                        next_index = (found + 1) % len(teleports)
                        next_tp = teleports[next_index]
                    else:
                        next_tp = teleports[0]
                elif teleports:
                    next_tp = teleports[0]
            if next_tp:
                r = next_tp.get('rect')
                if r:
                    try:
                        self.player.pixel_x = int(r.x)
                        self.player.pixel_y = int(r.y)
                    except Exception:
                        pass
        max_x = max(0, self.game_map.width *
                    self.game_map.tile_w - self.player.tile_w)
        max_y = max(0, self.game_map.height *
                    self.game_map.tile_h - self.player.tile_h)
        self.player.pixel_x = max(0, min(self.player.pixel_x, max_x))
        self.player.pixel_y = max(0, min(self.player.pixel_y, max_y))
        map_name = os.path.basename(tmx_file).lower()
        if map_name != "home_inn_1.tmx":
            self.slimes = spawn_slimes_randomly(self.game_map, count=8)
        else:
            self.slimes = []
        self.bosses = self.game_map.bosses
        self.towers = self.game_map.towers
        self.npcs = self.game_map.npcs
        if teleport_obj is not None:
            heal_amount = 20
            self.player.health = min(
                self.player.max_health, self.player.health + heal_amount)
            self.message = f"Teleported! Health +{heal_amount}"
            self.message_timer = 60
        try:
            self.camera.map_width = self.game_map.width * self.game_map.tile_w
            self.camera.map_height = self.game_map.height * self.game_map.tile_h
        except Exception:
            pass
        self.load_music(tmx_file)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.screen_width = self.screen.get_width()
            self.screen_height = self.screen.get_height()
        else:
            self.screen_width = self.default_width
            self.screen_height = self.default_height
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height))
        self.camera.update_screen_size(self.screen_width, self.screen_height)

    def handle_events(self):
        self.teleport_ready = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if not self.dialogue.active:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        world_x = mouse_x + self.camera.x
                        world_y = mouse_y + self.camera.y
                        projectile, is_crit = self.player.shoot_projectile(
                            world_x, world_y)
                        if projectile:
                            self.projectiles.append(projectile)
                            self.play_sound('projectile')
                            if is_crit:
                                self.floating_texts.append(FloatingText(
                                    self.player.pixel_x + self.player.tile_w // 2,
                                    self.player.pixel_y,
                                    "Critical!",
                                    (255, 0, 0)
                                ))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.debug_draw_teleports = not getattr(
                        self, 'debug_draw_teleports', False)
                    print(f"Debug draw teleports: {self.debug_draw_teleports}")
                    return
                if event.key == pygame.K_F11 or (event.key == pygame.K_RETURN and (pygame.key.get_mods() & pygame.KMOD_ALT)):
                    self.toggle_fullscreen()
                elif event.key == pygame.K_ESCAPE:
                    if self.fullscreen:
                        self.toggle_fullscreen()
                    else:
                        self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.dialogue.active:
                        self.dialogue.next()
                    else:
                        all_enemies = self.slimes + self.bosses + self.towers
                        if self.player.attack(all_enemies):
                            self.message = "Hit!"
                            self.message_timer = 30
                            self.play_sound('attacking')
                        else:
                            if self.player.stamina < self.player.attack_cost:
                                self.message = "Not enough stamina!"
                            elif self.player.attack_cooldown > 0:
                                self.message = "Attack on cooldown!"
                            else:
                                self.message = "No enemy in range!"
                            self.message_timer = 30
                elif event.key == pygame.K_e:
                    if self.nearby_npc and not self.dialogue.active:
                        self.dialogue.start_dialogue(self.nearby_npc.dialogues)
                        self.message = f"Talking to {self.nearby_npc.npc_name}..."
                        self.message_timer = 30
                    else:
                        p_rect = pygame.Rect(self.player.pixel_x, self.player.pixel_y,
                                             self.player.tile_w, self.player.tile_h)
                        for tp in getattr(self.game_map, 'teleports', []):
                            if tp.get('rect') and p_rect.colliderect(tp['rect']):
                                dest = tp.get('dest')
                                if dest:
                                    base_dir = os.path.dirname(os.path.abspath(
                                        self.current_map)) if self.current_map else os.path.dirname(os.path.abspath(__file__))
                                    dest_path = dest if os.path.isabs(
                                        dest) else os.path.join(base_dir, dest)
                                    if not os.path.exists(dest_path):
                                        alt = os.path.join(os.path.dirname(
                                            os.path.abspath(__file__)), 'map', dest)
                                        if os.path.exists(alt):
                                            dest_path = alt
                                    if os.path.exists(dest_path):
                                        self.load_map(dest_path, tp)
                                        self.teleport_cooldown = 30
                                        break

    def update(self):
        keys = pygame.key.get_pressed()
        self.player.handle_input(keys, self.game_map.collision_rects,
                                 self.game_map.width, self.game_map.height)
        self.player.update_combat()
        for slime in self.slimes:
            slime.update(self.player, self.game_map.collision_rects,
                         self.game_map.width, self.game_map.height, self)
        for boss in self.bosses:
            projectile = boss.update(self.player)
            if projectile:
                self.projectiles.append(projectile)
        for tower in self.towers:
            projectile = tower.update(self.player)
            if projectile:
                self.projectiles.append(projectile)
        self.nearby_npc = None
        for npc in self.npcs:
            if npc.can_interact(self.player):
                self.nearby_npc = npc
                break
        for proj in self.projectiles[:]:
            proj.update()
            if proj.is_enemy:
                player_rect = pygame.Rect(self.player.pixel_x, self.player.pixel_y,
                                          self.player.tile_w, self.player.tile_h)
                if proj.rect.colliderect(player_rect) and self.player.state != State.DEAD:
                    self.player.take_damage(proj.damage)
                    self.play_sound('taking_damage')
                    proj.active = False
            else:
                for slime in self.slimes:
                    if slime.state != State.DEAD:
                        slime_rect = pygame.Rect(slime.pixel_x, slime.pixel_y,
                                                 slime.tile_w, slime.tile_h)
                        if proj.rect.colliderect(slime_rect):
                            is_crit = random.random() < self.player.crit_chance
                            damage = proj.damage * \
                                self.player.crit_multiplier if is_crit else proj.damage
                            died, xp_reward = slime.take_damage(
                                damage, is_crit)
                            if died:
                                self.player.gain_xp(xp_reward, self)
                            if is_crit:
                                self.floating_texts.append(FloatingText(
                                    slime.pixel_x + slime.tile_w // 2,
                                    slime.pixel_y,
                                    "Critical!",
                                    (255, 0, 0)
                                ))
                            proj.active = False
                            break
                for boss in self.bosses:
                    if boss.state != State.DEAD:
                        boss_rect = pygame.Rect(boss.pixel_x, boss.pixel_y,
                                                boss.render_w, boss.render_h)
                        if proj.rect.colliderect(boss_rect):
                            is_crit = random.random() < self.player.crit_chance
                            damage = proj.damage * \
                                self.player.crit_multiplier if is_crit else proj.damage
                            died, xp_reward = boss.take_damage(
                                damage, is_crit)
                            if died:
                                self.player.gain_xp(xp_reward, self)
                            if is_crit:
                                self.floating_texts.append(FloatingText(
                                    boss.pixel_x + boss.render_w // 2,
                                    boss.pixel_y,
                                    "Critical!",
                                    (255, 0, 0)
                                ))
                            proj.active = False
                            break
                for tower in self.towers:
                    if tower.state != State.DEAD:
                        tower_rect = pygame.Rect(tower.pixel_x, tower.pixel_y,
                                                 tower.render_w, tower.render_h)
                        if proj.rect.colliderect(tower_rect):
                            is_crit = random.random() < self.player.crit_chance
                            damage = proj.damage * \
                                self.player.crit_multiplier if is_crit else proj.damage
                            died, xp_reward = tower.take_damage(
                                damage, is_crit)
                            if died:
                                self.player.gain_xp(xp_reward, self)
                            if is_crit:
                                self.floating_texts.append(FloatingText(
                                    tower.pixel_x + tower.render_w // 2,
                                    tower.pixel_y,
                                    "Critical!",
                                    (255, 0, 0)
                                ))
                            proj.active = False
                            break
            if not proj.active or proj.x < 0 or proj.x > self.game_map.width * self.game_map.tile_w or \
               proj.y < 0 or proj.y > self.game_map.height * self.game_map.tile_h:
                self.projectiles.remove(proj)
        for text in self.floating_texts[:]:
            text.update()
            if not text.is_alive():
                self.floating_texts.remove(text)
        self.camera.update(self.player.pixel_x, self.player.pixel_y,
                           self.player.tile_w, self.player.tile_h)
        self.teleport_ready = None
        if getattr(self, 'teleport_cooldown', 0) > 0:
            self.teleport_cooldown -= 1
        else:
            p_rect = pygame.Rect(self.player.pixel_x, self.player.pixel_y,
                                 self.player.tile_w, self.player.tile_h)
            for tp in getattr(self.game_map, 'teleports', []):
                if tp.get('rect') and p_rect.colliderect(tp['rect']):
                    self.teleport_ready = tp
                    break
        if self.message_timer > 0:
            self.message_timer -= 1
        all_dead = all(s.state == State.DEAD for s in self.slimes) and \
            all(b.state == State.DEAD for b in self.bosses) and \
            all(t.state == State.DEAD for t in self.towers)
        if all_dead and len(self.slimes + self.bosses + self.towers) > 0 and self.teleport_marker_timer == 0:
            tps = getattr(self.game_map, 'teleports', [])
            if tps:
                tp = tps[0]
                self.teleport_marker_rect = tp.get('rect')
                self.teleport_marker_timer = self.teleport_marker_duration
        if self.teleport_marker_timer > 0:
            self.teleport_marker_timer -= 1
            if self.teleport_marker_timer == 0:
                self.teleport_marker_rect = None

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.game_map.draw(self.screen, self.camera.x, self.camera.y)
        for npc in self.npcs:
            npc.draw(self.screen, self.camera.x, self.camera.y)
        for slime in self.slimes:
            slime.draw(self.screen, self.camera.x, self.camera.y)
        for boss in self.bosses:
            boss.draw(self.screen, self.camera.x, self.camera.y)
        for tower in self.towers:
            tower.draw(self.screen, self.camera.x, self.camera.y)
        self.player.draw(self.screen, self.camera.x, self.camera.y)
        for proj in self.projectiles:
            proj.draw(self.screen, self.camera.x, self.camera.y)
        for text in self.floating_texts:
            text.draw(self.screen, self.camera.x, self.camera.y)
        if getattr(self, 'debug_draw_teleports', False):
            for tp in getattr(self.game_map, 'teleports', []):
                try:
                    r = tp.get('rect')
                    if r:
                        sx = r.x - self.camera.x
                        sy = r.y - self.camera.y
                        pygame.draw.rect(
                            self.screen, (0, 255, 255), (sx, sy, r.width, r.height), 2)
                        lbl = self.font.render(
                            str(tp.get('dest')), True, (0, 255, 255))
                        self.screen.blit(lbl, (sx, sy - 18))
                except Exception:
                    pass
        if getattr(self, 'teleport_marker_rect', None) and getattr(self, 'teleport_marker_timer', 0) > 0:
            try:
                tp = self.teleport_marker_rect
                sx = int(tp.centerx - self.camera.x)
                sy = int(tp.top - self.camera.y) - 24
                pulse = 1.0 + 0.2 * \
                    (1 + math.sin(self.teleport_marker_timer * 0.2))
                arrow_h = int(16 * pulse)
                arrow_w = int(12 * pulse)
                points = [(sx, sy), (sx - arrow_w, sy + arrow_h),
                          (sx + arrow_w, sy + arrow_h)]
                pygame.draw.polygon(self.screen, (255, 215, 0), points)
                label = self.font.render("TELEPORT", True, (255, 215, 0))
                self.screen.blit(label, (sx - label.get_width() // 2, sy - 18))
            except Exception:
                pass
        draw_ui_bar(self.screen, 10, 10, 200, 25, self.player.health,
                    self.player.max_health, (46, 204, 113), (34, 139, 34), "Health")
        draw_ui_bar(self.screen, 10, 50, 200, 20, self.player.stamina,
                    self.player.max_stamina, (241, 196, 15), (150, 100, 0), "Stamina")
        draw_ui_bar(self.screen, 10, 85, 200, 15, self.player.xp,
                    self.player.xp_to_next_level, (138, 43, 226), (75, 0, 130), "XP")
        level_font = pygame.font.Font(None, 28)
        level_text = level_font.render(
            f"Level {self.player.level}", True, (255, 255, 255))
        level_bg = pygame.Surface(
            (level_text.get_width() + 10, level_text.get_height() + 4))
        level_bg.set_alpha(180)
        level_bg.fill((0, 0, 0))
        self.screen.blit(level_bg, (220, 10))
        self.screen.blit(level_text, (225, 12))
        stats_font = pygame.font.Font(None, 20)
        stats_y = 40
        stats_info = [
            f"DMG: {int(self.player.attack_damage)}",
            f"CRIT: {int(self.player.crit_chance * 100)}%"
        ]
        for stat_text in stats_info:
            stat_surf = stats_font.render(stat_text, True, (200, 200, 200))
            self.screen.blit(stat_surf, (225, stats_y))
            stats_y += 20
        controls = self.font.render(
            "WASD: Move | SHIFT: Run | SPACE: Attack | LMB: Shoot | E: Interact/Teleport", True, (255, 255, 255))
        self.screen.blit(controls, (10, self.screen_height - 30))
        if getattr(self, 'teleport_ready', None):
            prompt = self.font.render(
                "Press E to teleport", True, (0, 255, 255))
            self.screen.blit(prompt, (self.screen_width //
                             2 - prompt.get_width() // 2, 70))
        if self.nearby_npc and not self.dialogue.active:
            prompt = self.font.render(
                f"Press E to talk to {self.nearby_npc.npc_name}", True, (255, 255, 100))
            self.screen.blit(prompt, (self.screen_width //
                             2 - prompt.get_width() // 2, 90))
        if self.message_timer > 0:
            msg_surf = self.font.render(self.message, True, (255, 255, 0))
            self.screen.blit(msg_surf, (self.screen_width //
                             2 - msg_surf.get_width() // 2, 100))
        if self.player.state == State.DEAD:
            game_over_font = pygame.font.Font(None, 72)
            game_over_surf = game_over_font.render(
                "YOU DIED!", True, (255, 0, 0))
            self.screen.blit(game_over_surf, (self.screen_width // 2 - game_over_surf.get_width() // 2,
                                              self.screen_height // 2))
        self.dialogue.draw(self.screen, self.screen_width, self.screen_height)
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(100)
        pygame.quit()


# ======================================================================
#                             MAIN ENTRY
# ======================================================================

def main():
    root = tk.Tk()
    app = EnhancedFaceRecognitionSystem(root)
    root.mainloop()


if __name__ == "__main__":
    print("="*60)
    print("🚀 FACE LOCK + RPG GAME SYSTEM (One File)")
    print("="*60)
    print("✅ Tkinter дээр нүүр танилт")
    print("✅ Танилт амжилттай болсны дараа pygame тоглоом эхэлнэ")
    print("✅ dlib шаардлагагүй, зөвхөн OpenCV + pygame")
    print("="*60)
    print("\nПрограм эхэлж байна...\n")
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Програм зогссон")
    except Exception as e:
        print(f"\n❌ Алдаа: {e}")
        import traceback
        traceback.print_exc()
