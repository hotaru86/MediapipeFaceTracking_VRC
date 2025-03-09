"""
Copyright (c) 2025 hotaru86
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

import threading
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import ttk
from pythonosc import udp_client
import numpy as np
import json
import os

# JSONファイルのパス
PARAMS_FILE = "blendshape_params.json"

# JSONからパラメータを読み込む関数
def load_blendshape_params():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, "r") as f:
                params = json.load(f)
                print("パラメータをJSONから読み込みました")
                return params
        except Exception as e:
            print(f"JSONの読み込みに失敗しました: {e}")
    # ファイルがない場合や読み込み失敗の場合は既定値を返す
    print("既定のパラメータを使用します")
    return default_blendshape_params.copy()

# パラメータをJSONに保存する関数
def save_blendshape_params(params):
    try:
        with open(PARAMS_FILE, "w") as f:
            json.dump(params, f, indent=4)
        print("パラメータをJSONに保存しました")
    except Exception as e:
        print(f"パラメータの保存に失敗しました: {e}")

def update_and_save_params():
    for name, var_dict in blendshape_param_vars.items():
        # Entryウィジェットから各値を取得
        sensitivity = var_dict["sensitivity"].get()
        min_val = var_dict["min"].get()
        max_val = var_dict["max"].get()
        
        # 感度が0未満なら0に修正
        if sensitivity < 0:
            sensitivity = 0.0
            var_dict["sensitivity"].set(sensitivity)
        
        # 最小値が最大値以上の場合はデフォルト設定にする
        if min_val >= max_val:
            min_val = default_blendshape_params[name]["min"]
            max_val = default_blendshape_params[name]["max"]
            var_dict["min"].set(min_val)
            var_dict["max"].set(max_val)
        
        # blendshape_paramsを更新
        blendshape_params[name]["sensitivity"] = sensitivity
        blendshape_params[name]["min"] = min_val
        blendshape_params[name]["max"] = max_val

    # 更新後にJSONへ保存
    save_blendshape_params(blendshape_params)

#ARkit(mediapipe出力)のblendshape値をVRCFTのblendshape値に変換
def convert_blendshapes_dict_from_ARKit_to_VRCFT(source_dict):
    result_dict = dict()
    for name, score in source_dict.items():
        if name in name_mapping_0to1.keys():
            result_dict[name_mapping_0to1[name]] = score
        elif name in name_mapping_1to0.keys():
            result_dict[name_mapping_1to0[name]] = 1.0 - score
    result_dict["BrowExpressionRight"] = 0.5 * (source_dict["browInnerUp"] + source_dict["browOuterUpRight"]) - source_dict["browDownRight"]
    result_dict["BrowExpressionLeft"] = 0.5 * (source_dict["browInnerUp"] + source_dict["browOuterUpLeft"]) - source_dict["browDownLeft"]
    result_dict["EyeY"] = (source_dict["eyeLookUpLeft"] + source_dict["eyeLookUpRight"] - source_dict["eyeLookDownLeft"] - source_dict["eyeLookDownRight"]) / 4
    result_dict["EyeRightX"] = (source_dict["eyeLookOutRight"] - source_dict["eyeLookInRight"]) / 2
    result_dict["EyeLeftX"] = (source_dict["eyeLookInLeft"] - source_dict["eyeLookOutLeft"]) / 2
    result_dict["SmileFrownRight"] = source_dict["mouthSmileRight"] - source_dict["mouthFrownRight"]
    result_dict["SmileFrownLeft"] = source_dict["mouthSmileLeft"] - source_dict["mouthFrownLeft"]
    result_dict["JawX"] = source_dict["jawRight"] - source_dict["jawLeft"]
    result_dict["MouthX"] = source_dict["mouthRight"] - source_dict["mouthLeft"]
    result_dict["CheekPuffLeft"] = source_dict["cheekPuff"]
    result_dict["CheekPuffRight"] = source_dict["cheekPuff"]
    return result_dict

def float_to_3bit_binary(value):
    if not (0.0 <= value <= 1.0):
        value = np.clip(value, 0.0, 1.0)
    int_value = round(value * 7)
    return format(int_value, '03b')

def send_vrcft_blendshapes_osc(face_blendshapes):
    names = [cat.category_name for cat in face_blendshapes]
    scores = [cat.score for cat in face_blendshapes]
    renamed = convert_blendshapes_dict_from_ARKit_to_VRCFT(dict(zip(names, scores)))
    mapped_score_dict = dict()
    for name, score in renamed.items():
        try:
            sensitivity =  blendshape_params[name]["sensitivity"]
            min = blendshape_params[name]["min"]
            max = blendshape_params[name]["max"]
            #blendshape値を感度倍した後、パラメータで設定した最小-最大の区間にマッピングする
            if name in name_range_0to1:
                renamed[name] = np.clip(score * sensitivity, 0.0, 1.0)
                mapped_score_dict[name] = min + (renamed[name] - 0.0) * (max - min) / (1.0 - 0.0)
            elif name in name_range_1to1:
                renamed[name] = np.clip(score * sensitivity, -1.0, 1.0)
                mapped_score_dict[name] = min + (renamed[name] - (-1.0)) * (max - min) / (1 - (-1.0))
        
        except Exception as e:
            print(f"Error: {e}")
            if name in name_range_1to1:
                mapped_score_dict[name] = np.clip(mapped_score_dict[name], -1.0, 1.0)
            else:
                mapped_score_dict[name] = np.clip(mapped_score_dict[name], 0.0, 1.0)
    for name, mapped_score in mapped_score_dict.items():
        prefix = "/avatar/parameters/FT/v2/"
        client.send_message(f"{prefix}{name}", mapped_score)
        if name in name_range_1to1:
            client.send_message(f"{prefix}{name}Negative", bool(mapped_score < 0))
        for i in range(3):
            client.send_message(f"{prefix}{name}{2**i}", bool(float_to_3bit_binary(abs(mapped_score))[i] == "1"))

# マッピング定義
name_mapping_0to1 = {
    "eyeSquintLeft": "EyeSquintLeft",
    "eyeSquintRight": "EyeSquintRight",
    "cheekPuff": "CheekPuffLeft", #VRCFTは左右別だが、
    "jawOpen": "JawOpen",
    "mouthClose": "MouthClosed",
    "jawForward": "JawForward",
    "mouthRollUpper": "LipSuckUpper",
    "mouthRollLower": "LipSuckLower",
    "mouthFunnel": "LipFunnel",
    "mouthPucker": "LipPucker",
    "mouthStretchLeft": "MouthStretchLeft",
    "mouthStretchRight": "MouthStretchRight",
    "mouthPressLeft": "MouthPress",
}
name_mapping_1to0 = {
    "eyeBlinkLeft": "EyeLidLeft",
    "eyeBlinkRight": "EyeLidRight",
}

name_range_0to1 = set(name_mapping_0to1.values()) | set(name_mapping_1to0.values()) | {"CheekPuffLeft", "CheekPuffRight"}
name_range_1to1 = {
    "EyeLeftX", "EyeRightX", "EyeY",
    "BrowExpressionLeft", "BrowExpressionRight",
    "JawX", "MouthX",
    "SmileFrownLeft", "SmileFrownRight",
}
name_not_modify_norm_range = {
    "EyeLeftX", "EyeRightX", "EyeY",
}

default_blendshape_params = {
    "EyeLeftX": {"sensitivity": 1.0, "max": 0.5, "min": -0.5},
    "EyeRightX": {"sensitivity": 1.0, "max": 0.5, "min": -0.5},
    "EyeY": {"sensitivity": 0.5, "max": 0.0, "min": -1.0},
    "EyeLidLeft": {"sensitivity": 1.0, "max": 1.0, "min": -1.0},
    "EyeLidRight": {"sensitivity": 1.0, "max": 1.0, "min": -1.0},
    "EyeSquintLeft": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "EyeSquintRight": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "BrowExpressionLeft": {"sensitivity": 1.0, "max": 1.0, "min": -1.0},
    "BrowExpressionRight": {"sensitivity": 1.0, "max": 1.0, "min": -1.0},
    "NoseSneer": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "CheekPuffLeft": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "CheekPuffRight": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "JawOpen": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "MouthClosed": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "JawX": {"sensitivity": 1.0, "max": 1.0, "min": -1.0},
    "JawForward": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "LipSuckUpper": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "LipSuckLower": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "LipFunnel": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "LipPucker": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "MouthUpperUp": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "MouthLowerDown": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "MouthX": {"sensitivity": 1.0, "max": 1.0, "min": -1.0},
    "SmileFrownLeft": {"sensitivity": 1.0, "max": 1.0, "min": -1.0},
    "SmileFrownRight": {"sensitivity": 1.0, "max": 1.0, "min": -1.0},
    "MouthStretchLeft": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "MouthStretchRight": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "MouthRaiserLeft": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "MouthRaiserRight": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "MouthPress": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
    "TongueOut": {"sensitivity": 1.0, "max": 1.0, "min": 0.0},
}

# mediapipeのセットアップ
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# OSCクライアント（後で更新するため、グローバル変数とする）
client = None

# プログラム開始時にJSONから読み込み
blendshape_params = load_blendshape_params()

# ------------------------- キャプチャ処理 -------------------------
# 利用可能なカメラデバイスをスキャン
def list_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# 使用するカメラデバイスの選択
def update_selected_camera(event=None):
    global selected_camera
    selected_camera = int(camera_combobox.get())

capture_running = False
capture_thread = None

def capture_loop():
    global capture_running, client
    if selected_camera is None:
        print("カメラが選択されていません")
        return

    cap = cv2.VideoCapture(selected_camera)
    if not cap.isOpened():
        print("カメラが開けませんでした")
        return

    while capture_running:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        cv2.imshow('Face Landmarks', frame)

        if detection_result.face_blendshapes:
            send_vrcft_blendshapes_osc(detection_result.face_blendshapes[0])

        time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------- GUI部分 -------------------------
def start_capture():
    global capture_running, capture_thread, client
    # OSC設定をGUIから取得
    ip_val = ip_entry.get().strip()
    try:
        port_val = int(port_entry.get().strip())
    except ValueError:
        print("正しいポート番号を入力してください")
        return
    # OSCクライアントの更新
    client = udp_client.SimpleUDPClient(ip_val, port_val)
    # キャプチャ開始前に設定入力欄を無効化
    ip_entry.config(state="disabled")
    port_entry.config(state="disabled")
    camera_combobox.config(state="disabled")
    # キャプチャ開始
    if not capture_running:
        capture_running = True
        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()
        status_label.config(text="キャプチャ中")
    else:
        print("既にキャプチャは動作中です")

def stop_capture():
    global capture_running, capture_thread
    capture_running = False
    if capture_thread is not None:
        capture_thread.join()
    # キャプチャ停止後、設定入力欄を再度有効化
    ip_entry.config(state="normal")
    port_entry.config(state="normal")
    camera_combobox.config(state="normal")
    status_label.config(text="停止中")

# TkEasyGUI（Tkinter）のウィンドウ作成
root = tk.Tk()
root.title("Face Tracking OSC Sender")

#左フレーム
left_frame = ttk.Frame(root)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

# OSC設定フレーム
osc_frame = ttk.LabelFrame(left_frame, text="OSC設定")
osc_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

ttk.Label(osc_frame, text="IP:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
ip_entry = ttk.Entry(osc_frame)
ip_entry.insert(0, "127.0.0.1")
ip_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(osc_frame, text="Port:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
port_entry = ttk.Entry(osc_frame)
port_entry.insert(0, "9000")
port_entry.grid(row=1, column=1, padx=5, pady=5)

#カメラ設定用フレーム
camera_frame = ttk.LabelFrame(left_frame, text="カメラ設定")
camera_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

ttk.Label(camera_frame, text="カメラ選択:").grid(row=0, column=0, sticky="e", padx=5, pady=5)

# カメラリストの取得とプルダウン作成
camera_list = list_cameras()
camera_combobox = ttk.Combobox(camera_frame, values=camera_list, state="readonly")
camera_combobox.grid(row=0, column=1, padx=5, pady=5)
camera_combobox.bind("<<ComboboxSelected>>", update_selected_camera)

# デフォルトカメラを設定
if camera_list:
    camera_combobox.current(0)
    selected_camera = camera_list[0]
else:
    selected_camera = None


# 操作ボタンフレーム
control_frame = ttk.LabelFrame(left_frame, text="操作")
control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

start_button = ttk.Button(control_frame, text="キャプチャ開始", command=start_capture)
start_button.grid(row=0, column=0, padx=5, pady=5)

stop_button = ttk.Button(control_frame, text="キャプチャ停止", command=stop_capture)
stop_button.grid(row=0, column=1, padx=5, pady=5)

status_label = ttk.Label(root, text="停止中")
status_label.grid(row=2, column=0, padx=10, pady=10)

# 右フレーム
right_frame = ttk.Frame(root)
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# パラメータ設定用のスクロール領域作成
scroll_container = ttk.Frame(right_frame)
scroll_container.grid(row=0, column=0, sticky="nsew")

# Canvasとスクロールバーを作成
canvas = tk.Canvas(scroll_container, width=300, height=300)
canvas.grid(row=0, column=0, sticky="nsew")

scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
scrollbar.grid(row=0, column=1, sticky="ns")
canvas.configure(yscrollcommand=scrollbar.set)

# Canvas内にスクロール可能なフレームを作成
parameters_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=parameters_frame, anchor="nw")

# スクロール領域の更新設定
def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
parameters_frame.bind("<Configure>", on_configure)

#blendshapeのパラメータ設定
param_row = 0
ttk.Label(parameters_frame, text="Blendshapeごとの設定").grid(row=param_row, column=0, columnspan=2, padx=5, pady=5, sticky="w")
ttk.Button(parameters_frame, text="適用", command=lambda: update_and_save_params()).grid(row=param_row, column=2, columnspan=2, padx=5, pady=5, sticky="w")
param_row+=1
ttk.Label(parameters_frame, text="感度").grid(row=param_row, column=1, padx=5, pady=5, sticky="w")
ttk.Label(parameters_frame, text="最小値").grid(row=param_row, column=2, padx=5, pady=5, sticky="w")
ttk.Label(parameters_frame, text="最大値").grid(row=param_row, column=3, padx=5, pady=5, sticky="w")
param_row+=1
blendshape_param_vars = {}  # 各Entryの変数を格納
for i, (name, params) in enumerate(blendshape_params.items(), 1):
        ttk.Label(parameters_frame, text=name).grid(row=i+param_row, column=0, padx=5, pady=5, sticky="w")
        
        # 感度
        sensitivity_var = tk.DoubleVar(value=params["sensitivity"])
        sensitivity_entry = ttk.Entry(parameters_frame, textvariable=sensitivity_var, width=6)
        sensitivity_entry.grid(row=i+param_row, column=1, padx=5, pady=5)
        
        # 最小値
        min_var = tk.DoubleVar(value=params["min"])
        min_entry = ttk.Entry(parameters_frame, textvariable=min_var, width=6)
        min_entry.grid(row=i+param_row, column=2, padx=5, pady=5)

        # 最大値
        max_var = tk.DoubleVar(value=params["max"])
        max_entry = ttk.Entry(parameters_frame, textvariable=max_var, width=6)
        max_entry.grid(row=i+param_row, column=3, padx=5, pady=5)

        # 変数を辞書に登録
        blendshape_param_vars[name] = {"sensitivity": sensitivity_var, "min": min_var, "max": max_var}


# 列の幅を自動調整（必要に応じて）
parameters_frame.columnconfigure(1, weight=1)

# GUIのメインループ開始
root.mainloop()
