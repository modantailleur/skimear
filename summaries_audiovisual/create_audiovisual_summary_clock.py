import numpy as np
import os
import math
import librosa
import imageio
import subprocess
from datetime import timedelta
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
import soundfile as sf
import sys
import argparse

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ==========
# Timeline
# ==========
def convert_time_to_float(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours + minutes / 60.0

def convert_float_to_time(decimal_hours):
    decimal_hours = decimal_hours % 24
    hours = int(decimal_hours)
    minutes = int((decimal_hours - hours) * 60)
    return '{:02}:{:02}'.format(hours, minutes)


def main(args):
    # ==========
    # Parameters
    # ==========
    hop = 0.1  # frame step in sec
    flen = 1
    audio_file = args.input
    if args.output is None:
        output_video_path = "./summaries_audiovisual/video/" + os.path.splitext(os.path.basename(audio_file))[0] + ".mp4"
    else:
        output_video_path = args.output
        if not output_video_path.lower().endswith(".mp4"):
            raise ValueError("Extension has to be .mp4 for the output video path")
    
    frame_rate = 1 / hop
    waveform, sr = librosa.load(audio_file, sr=None, mono=True)
    run_time_linear = True


    if os.path.exists(audio_file[:-4] + ".csv"):
        df_timeline = pd.read_csv(audio_file[:-4] + ".csv")
        # df_timeline['period'] = df_timeline['period'].fillna(df_timeline['clusters_datetimes'])
        df_timeline['period'] = pd.to_datetime(df_timeline['period'])

        time_difference = df_timeline.loc[df_timeline.index[-1], 'period'] - df_timeline.loc[0, 'period']
        time_difference = str(time_difference).split(' ')[-1]
        time_difference = convert_time_to_float(time_difference[:-3])

        max_time = convert_time_to_float(df_timeline.loc[df_timeline.index[-1], 'period'].strftime('%H:%M'))
        df_timeline.loc[1, 'period'] = df_timeline.loc[0, 'period']
        df_timeline = df_timeline.drop(df_timeline.index[0]).reset_index(drop=True)
        df_timeline = df_timeline.drop(df_timeline.index[-1]).reset_index(drop=True)
        df_timeline['idx_seconds'] = df_timeline.index * timedelta(seconds=5)
        df_timeline['idx_seconds'] = df_timeline['idx_seconds'].apply(lambda x: str(x).split(' ')[-1])
        df_timeline['period'] = df_timeline['period'].dt.strftime('%H:%M')

        label_datetimes = {label: idx for label, idx in zip(df_timeline["period"].values, df_timeline["idx_seconds"].values)}
    else:
        label_datetimes = {}
        print('No timeline file found. Clock will not move.')

    def make_timeline_frame(t, label_datetimes):
        seconds_list = []
        keys = label_datetimes.keys()

        hours_list = [convert_time_to_float(key) for key in keys]
        hours_original_list = [elem for elem in hours_list]
        hours_list.append(max_time)
        hours_list = np.array([(hours_list[i] - hours_list[i-1]) % 24 for i in range(len(hours_list))])
        hours_list = np.cumsum(hours_list[1:])
        hours_list = [0] + list(hours_list)
        hours_list = np.array(hours_list)
        hours_list = [elem + hours_original_list[0] for elem in hours_list]

        for _, ((label, label_datetime), x) in enumerate(zip(label_datetimes.items(), hours_list)):
            hours, minutes, seconds = map(int, label_datetime.split(':'))
            seconds = hours * 3600 + minutes * 60 + seconds
            seconds_list.append(seconds)
            seconds = seconds * time_difference / duration
        seconds_list = np.array(seconds_list)

        current_seconds = (t % duration)
        projected_seconds = seconds_list
        idx = np.searchsorted(projected_seconds, current_seconds)

        if (idx != 0) & (idx != len(projected_seconds)):
            bef_sec = projected_seconds[idx-1]
            aft_sec = projected_seconds[idx]
            bef_hour = hours_list[idx-1]
            aft_hour = hours_list[idx]
        elif idx == len(projected_seconds):
            bef_sec = projected_seconds[idx-1]
            aft_sec = duration
            bef_hour = hours_list[idx-1]
            aft_hour = hours_list[idx]
        else:
            bef_sec = 0
            aft_sec = projected_seconds[idx]
            bef_hour = 0
            aft_hour = hours_list[idx]

        if aft_sec != bef_sec:
            ratio_bef = (aft_sec - current_seconds) / (aft_sec - bef_sec)
            ratio_aft = (current_seconds - bef_sec) / (aft_sec - bef_sec)
            time_warped = bef_hour * ratio_bef + aft_hour * ratio_aft
        else:
            time_warped = aft_hour
        return time_warped

    # ==========
    # Frame generator
    # ==========
    def make_frame(t):
        if run_time_linear or not label_datetimes:
            # Force linear mapping (0–24h evenly over audio duration)
            time_float = (t / duration) * 24
        else:
            # Use warped timeline
            time_float = make_timeline_frame(t, label_datetimes)

        width, height = 1280, 720
        frame = Image.new('RGBA', (width, height), (0, 0, 0, 255))
        d = ImageDraw.Draw(frame)

        # big clock parameters
        clock_radius = 200
        clock_center = (width // 2, height // 2)
        d.ellipse((clock_center[0] - clock_radius, clock_center[1] - clock_radius,
                clock_center[0] + clock_radius, clock_center[1] + clock_radius),
                fill=(0, 0, 0, 255), outline=(255, 255, 255, 255), width=8)

        # ===== Draw 8 ticks =====
        for i in range(8):
            angle = math.radians(i * 45) - math.pi/2
            tick_start = clock_radius * 0.85
            tick_end = clock_radius * 1.0
            x1 = clock_center[0] + tick_start * math.cos(angle)
            y1 = clock_center[1] + tick_start * math.sin(angle)
            x2 = clock_center[0] + tick_end * math.cos(angle)
            y2 = clock_center[1] + tick_end * math.sin(angle)
            d.line((x1, y1, x2, y2), fill=(255, 255, 255, 255), width=6)

        # ===== Clock hand =====
        angle = (time_float % 24 / 24.0) * 360.0
        angle_rad = math.radians(angle)
        hand_length = clock_radius * 0.8
        hand_x = clock_center[0] + hand_length * math.cos(angle_rad - math.pi/2)
        hand_y = clock_center[1] + hand_length * math.sin(angle_rad - math.pi/2)
        d.line((clock_center, (hand_x, hand_y)), fill=(255, 255, 255, 255), width=10)

        # ===== Labels =====
        font_path = "./fonts/00TT.TTF"
        font_clock = ImageFont.truetype(font_path, 50)
        hour_labels = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
        for i, label in enumerate(hour_labels):
            label_angle = math.radians(i * 45) - math.pi/2
            # Distance factor: 1.3 (top/bottom) → 1.5 (sides)
            horizontal_strength = abs(math.cos(label_angle))  # 0 for vertical, 1 for horizontal
            label_distance = clock_radius * (1.15 + 0.25 * horizontal_strength)
            x = clock_center[0] + label_distance * math.cos(label_angle)
            y = clock_center[1] + label_distance * math.sin(label_angle)
            d.text((x, y), label, fill=(255, 255, 255, 255), font=font_clock, anchor="mm")

        return np.array(frame)

    # ==========
    # Write video
    # ==========
    duration = len(waveform)/sr
    temp_output_video_path = "./summaries_audiovisual/video/.temp/.temp.mp4"
    os.makedirs(os.path.dirname(temp_output_video_path), exist_ok=True)
    if os.path.exists(temp_output_video_path):
        os.remove(temp_output_video_path)

    temp_audio_path = "./summaries_audiovisual/video/.temp/.temp.wav"
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    sf.write(temp_audio_path, librosa.util.normalize(waveform), sr)

    writer = imageio.get_writer(temp_output_video_path, format='ffmpeg', mode='I', fps=int(frame_rate))
    for t in range(int(duration * frame_rate)):
        frame = make_frame(t / frame_rate)
        writer.append_data(frame)
        print(f"Progress: {t+1}/{int(duration * frame_rate)}", end='\r')
    writer.close()

    output_video_path = "./summaries_audiovisual/video/" + os.path.splitext(os.path.basename(audio_file))[0] + "_clock.mp4"
    command = [
        'ffmpeg', '-y', '-i', temp_output_video_path, '-i', temp_audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
        '-map', '0:v:0', '-map', '1:a:0', output_video_path
    ]
    subprocess.run(command)
    print(f"Video saved as {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video for your audio summary.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Optional path to save the output video. Extension has to be .mp4. By default, " \
                        "the output path is set to './summaries_audiovisual/video/' folder, with the same name as your input audio file but as a '.mp4'")
    args = parser.parse_args()

    main(args)