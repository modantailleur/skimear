import numpy as np
import random
import os
from datetime import timedelta
from PIL import ImageFont, ImageDraw, Image, ImageFilter
import imageio
import subprocess
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import torch.nn.functional as F
import cv2
import soundfile as sf
import sys
import argparse

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pann.pann_inference import PANNsModel
import utils.util as ut

def convert_time_to_float(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours + minutes / 60.0

def convert_float_to_time(decimal_hours):
    # Split the decimal hours into hours and minutes
    decimal_hours = decimal_hours % 24
    if np.isnan(decimal_hours) == False:
        hours = int(decimal_hours)
        minutes = int((decimal_hours - hours) * 60)
        out = '{:02}:{:02}'.format(hours, minutes)
        # Format the hours and minutes as a string in HH:MM format
    else:
        out = "NaN"
    return out

def sort_labels(df, scores):
    scores = scores[df['index'].values]
    labels = df['class'].values
    score_label_tuples = list(zip(scores, labels))
    # sort the tuples based on the score in descending order
    sorted_tuples = sorted(score_label_tuples, reverse=True)

    # extract the sorted labels from the sorted tuples
    sorted_labels = [t[1] for t in sorted_tuples]
    sorted_scores = [t[0] for t in sorted_tuples]

    return(sorted_scores, sorted_labels)

# Just a utility function to plot two rectangles
def plot_rectangles(rect1, rect2):
    """Plot two rectangles with their expanded versions."""
    BL1, BR1, TL1, TR1 = rect1
    BL2, BR2, TL2, TR2 = rect2

    fig, ax = plt.subplots(1)
    
    # Original rectangles
    plot_rectangle(ax, rect1, 'blue', 'Rectangle 1')
    plot_rectangle(ax, rect2, 'green', 'Rectangle 2')

    # Set the limits of the plot
    all_points = TL1 + TR1 + BL1 + BR1 + TL2 + TR2 + BL2 + BR2
    x_coords = all_points[::2]
    y_coords = all_points[1::2]
    ax.set_xlim(min(x_coords), max(x_coords))
    ax.set_ylim(min(y_coords), max(y_coords))

    # Add legend and show plot
    ax.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Just a utility function to plot a single rectangle
def plot_rectangle(ax, rect, color='blue', label='Rectangle'):
    """Plot a single rectangle on the given axes."""
    TL, TR, BL, BR = rect
    width = TR[0] - TL[0]
    height = BL[1] - TL[1]
    ax.add_patch(patches.Rectangle((TL[0], TL[1]), width, height, linewidth=1, edgecolor=color, facecolor='none', label=label))

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main(args):
    # Set the random seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    hop = 0.1  # Time in seconds between each frame
    flen = 1  # Length of each frame in seconds

    audio_file = args.input

    if args.output is None:
        output_video_path = "./summaries_audiovisual/video/" + os.path.splitext(os.path.basename(audio_file))[0] + ".mp4"
    else:
        output_video_path = args.output
        if not output_video_path.lower().endswith(".mp4"):
            raise ValueError("Extension has to be .mp4 for the output video path")

    waveform, sr = librosa.load(audio_file, sr=32000)
    # Normalize audio
    audio = librosa.util.normalize(waveform)
    # Chunks the audio into overlapping frames
    chunker = ut.AudioChunks(n=round(sr * flen), hop=round(sr * hop))
    audio_n = chunker.chunks_with_hop(audio)

    # Initialize and load PANNs model
    pann_model = PANNsModel()
    pann_model.load_model()

    # Get embeddings and logits from PANNs model
    logit = pann_model.get_logits(audio_n)

    #if using cnn14declev model (with a decision every 10ms)
    if len(logit.shape) == 3:
        target_fps = 12
        logit = logit.reshape(logit.shape[0]*logit.shape[1], -1)
        logit = logit.unsqueeze(0).unsqueeze(0)
        logit = F.interpolate(logit, size=(int(target_fps*len(audio)/sr), 527), mode='bilinear', align_corners=False)
        logit = logit.squeeze(0).squeeze(0) 
        frame_rate = logit.shape[0]/(len(audio)/sr)
    else:
        frame_rate = 1/hop

    # Extract top labels sorted by score
    file_classes_path = './utils/audioset_for_video.xlsx'
    df_classes = pd.read_excel(file_classes_path)
    df_classes = df_classes.dropna(subset=['keep'])

    # Level is the level of the class (air, ground, background)
    level_dict = {row['class']: row['Level'] if pd.notna(row['Level']) else None for _, row in df_classes.iterrows()}
    # Type is the type of the class (repeating, vanishing)
    type_dict = {row['class']: row['Type'] if pd.notna(row['Type']) else None for _, row in df_classes.iterrows()}

    #####################
    ### FOREGROUND LABELS

    # take top2 of labels
    top = 2
    labels = [sort_labels(df_classes[df_classes['Level']!='b'], log.cpu().numpy())[1][:top] for log in logit]
    scores = [sort_labels(df_classes[df_classes['Level']!='b'], log.cpu().numpy())[0][:top] for log in logit]

    #filter predictions that are extremely low
    threshold = 0.10
    labels = [[label for label, score in zip(lab, sco) if score > threshold] for lab, sco in zip(labels, scores)]
    scores = [[score for score in sco if score > threshold] for sco in scores]

    ####################
    ### BACKGROUND LABELS

    #take top2 of labels
    top = 2
    labels_bg = [sort_labels(df_classes[df_classes['Level']=='b'], log.cpu().numpy())[1][:top] for log in logit]
    scores_bg = [sort_labels(df_classes[df_classes['Level']=='b'], log.cpu().numpy())[0][:top] for log in logit]

    #filter predictions that are extremely low
    threshold = 0.01
    labels_bg = [[label for label, score in zip(lab, sco) if score > threshold] for lab, sco in zip(labels_bg, scores_bg)]
    scores_bg = [[score for score in sco if score > threshold] for sco in scores_bg]

    ##############
    ## TIMELINE DISPLAY

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

        # Convert timestamp strings to datetime objects
        label_datetimes = {label: idx for label, idx in zip(df_timeline["period"].values, df_timeline["idx_seconds"].values)}
    else:
        label_datetimes = {}
        print('No timeline file found. Please provide a timeline file with the same name as the wav file to have a clock on the video.')

    ###########
    #########

    def is_far_enough(x, y, textlabels=None, min_distance=50, text_size=None):
        if textlabels is None:
            return True
        else:
            for pos_x, pos_y, pos_size in [(textlabel.pos_x, textlabel.pos_y, textlabel.text_size) for textlabel in textlabels]:
                pos = (pos_x, pos_y)
                if text_size is None:
                    if np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2) < min_distance:
                        return False
                if text_size is not None:
                    rect1 = [[x, y], [x + text_size[0], y], [x, y + text_size[1]], [x + text_size[0], y + text_size[1]]]
                    rect2 = [[pos[0], pos[1]], [pos[0] + pos_size[0], pos[1]], [pos[0], pos[1] + pos_size[1]], [pos[0] + pos_size[0], pos[1] + pos_size[1]]]
                    if are_rectangles_overlapping(rect1, rect2, min_distance=min_distance):
                        return False
            return True

    def are_rectangles_overlapping(rect1, rect2, min_distance=50):
        BL1, BR1, TL1, TR1 = rect1
        BL2, BR2, TL2, TR2 = rect2

        # # Plot the rectangles
        # plot_rectangles(rect1, rect2, min_distance)
        
        TL1[0] = TL1[0] - min_distance
        TL1[1] = TL1[1] + min_distance
        TR1[0] = TR1[0] + min_distance
        TR1[1] = TR1[1] + min_distance
        BL1[0] = BL1[0] - min_distance
        BL1[1] = BL1[1] - min_distance
        BR1[0] = BR1[0] + min_distance
        BR1[1] = BR1[1] - min_distance

        # # Plot the rectangles
        # plot_rectangles(rect1, rect2, min_distance)

        # if rectangle has area 0, no overlap
        if TL1[0] == BR1[0] or TL1[1] == BR1[1] or BR2[0] == TL2[0] or TL2[1] == BR2[1]:
            return False
        
        # If one rectangle is on left side of other
        if TL1[0] > BR2[0] or TL2[0] > BR1[0]:
            return False

        # If one rectangle is above other
        if BR1[1] > TL2[1] or BR2[1] > TL1[1]:
            return False
        
        return True

    # Define a function to generate random positions within a frame
    def random_position(x_range, y_range, text_size=None, textlabels=None, level=None, min_distance=200, max_iter=300, variance_level=4):
        """
        Parameters
        ----------
        x_range : tuple
            The range of x values to generate random positions in.
        y_range : tuple
            The range of y values to generate random positions in.
        text_size : tuple
            The size of the text to be placed at the random position.
        textlabels : list
            A list of TextLabel objects that have already been placed.
        level : str
            The level of the text label (air, ground, background).
        min_distance : int
            The minimum distance between the new textlabel position and the existing textlabels.
        max_iter : int
            The maximum number of iterations to try generating a random position.
        variance_level : int
            The variance level for the normal distribution used to generate random positions.
        """

        for k in range(max_iter):
            if level is None:
                # x = random.randint(x_range[0], x_range[1])
                x = int(random.normalvariate((x_range[0] + x_range[1]) / 2, (x_range[1] - x_range[0]) / variance_level))
                # y = random.randint(y_range[0], y_range[1])
                y = int(random.normalvariate((y_range[0] + y_range[1]) / 2, (y_range[1] - y_range[0]) / variance_level))
            else:
                # if background
                if level == "b":
                    y_range0 = y_range[0] + (y_range[1]-y_range[0]) * 0.80
                    y_range1 = y_range[1]
                # if ground
                elif level == "g":
                    y_range0 = y_range[0] + (y_range[1]-y_range[0]) * 0.25
                    y_range1 = y_range[0] + (y_range[1]-y_range[0]) * 0.75
                # if air
                elif level == "a":
                    y_range0 = y_range[0]
                    y_range1 = y_range[0] + (y_range[1]-y_range[0]) * 0.20
                else:
                    y_range0 = y_range[0]
                    y_range1 = y_range[1]

                # x = random.randint(x_range[0], x_range[1])
                x = int(random.normalvariate((x_range[0] + x_range[1]) / 2, (x_range[1] - x_range[0]) / variance_level))
                # y = random.randint(int(y_range0), int(y_range1))
                y = int(random.normalvariate((y_range0 + y_range1) / 2, (y_range1 - y_range0) / variance_level))

            if is_far_enough(x, y, textlabels=textlabels, min_distance=min_distance, text_size=text_size):
                return x, y
        return(x, y)

    def make_timeline_frame(t, label_datetimes):
        """
        This function returns the real-time timestamp based on the video timestamp.
        
        Parameters  
        ----------
        t : float
            The current time in seconds.
        label_datetimes : dict
            A dictionary containing the timestamps of the labels.
        """
        # Plot labels at specified timestamps
        seconds_list = []
        keys = label_datetimes.keys()

        # hours_list = np.array([int(key[:-1]) if key[:-1].isdigit() else int(key[:-2]) for key in keys])
        hours_list = [convert_time_to_float(key) for key in keys]
        hours_original_list = [elem for elem in hours_list]
        hours_list.append(max_time)
        hours_list = np.array([(hours_list[i] - hours_list[i-1]) % 24 for i in range(len(hours_list))])
        hours_list = hours_list[1:]
        hours_list = np.cumsum(hours_list)
        hours_list = [0] + list(hours_list)
        hours_list = np.array(hours_list)
        hours_list = [elem+hours_original_list[0] for elem in hours_list]

        for _, ((label, label_datetime), x) in enumerate(zip(label_datetimes.items(), hours_list)):
            hours, minutes, seconds = map(int, label_datetime.split(':'))
            seconds = hours * 3600 + minutes * 60 + seconds
            seconds_list.append(seconds)
            seconds = seconds * time_difference / duration
        seconds_list = np.array(seconds_list)

        # Plot the current position with a yellow cross
        current_seconds = (t % duration)
        projected_seconds = seconds_list

        idx = np.searchsorted(projected_seconds, current_seconds)
        
        if (idx != 0) & (idx != len(projected_seconds)):
            bef_sec = projected_seconds[idx-1]
            aft_sec = projected_seconds[idx]
            bef_hour = hours_list[idx-1]
            aft_hour = hours_list[idx]
        if idx == len(projected_seconds):
            bef_sec = projected_seconds[idx-1]
            aft_sec = duration
            bef_hour = hours_list[idx-1]
            aft_hour = hours_list[idx] 
        if idx == 0:
            bef_sec = 0
            aft_sec = projected_seconds[idx]
            bef_hour = 0
            aft_hour = hours_list[idx]

        if aft_sec != bef_sec:
            ratio_bef = (aft_sec - current_seconds) / (aft_sec - bef_sec)
            ratio_aft = (current_seconds - bef_sec) / (aft_sec - bef_sec)
            time_warped = bef_hour * ratio_bef + aft_hour * ratio_aft
        else:
            #first index
            time_warped = aft_hour

        return time_warped

    def time_sine_wave(x):
        if x < 0 or x > 24:
            return 0  # Outside the defined range, return 0
        else:
            return 0.5+0.5*np.cos(np.pi*x/12)
        
    def adjust_alpha(time_float, min_value=0.3, max_value=0.8):
        time_float = time_float % 24
        return min_value + time_sine_wave(time_float) * (max_value - min_value) 

    def getTextSize(text, font_path, font_size=32, font_scale=1, font_thickness=2):
        # Load the font
        font = ImageFont.truetype(font_path, int(font_size*font_scale))

        # Create a new image with a transparent background
        dummy_image = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy_image)

        # Get the bounding box of the text
        bbox = draw.textbbox((0, 0), text, font=font)

        # Calculate width and height from the bounding box
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        return (width, height)

    class TextLabel:
        """
        Class to store text labels and their properties.
        Parameters
        ----------
        text : str
            The text to display.
        t : float
            The time at which the text label was created.
        pos : tuple
            The position of the text label.
        type : str
            The type of the text label (repeating, vanishing).
        font : int
            The font to use for the text label.
        font_size : int
            The font size to use for the text label.
        font_scale : float
            The font scale to use for the text label.
        font_thickness : int
            The font thickness to use for the text label.
        color : tuple
            The color of the text label.
        alpha : float
            The alpha value for transparency of the text label.
        """
        def __init__(self, text, t, pos=(None, None), type=None, font=cv2.FONT_HERSHEY_COMPLEX, font_size=32, font_scale=1, font_thickness=2, color=(255, 255, 255), alpha=1.0):
            self.pos_x = pos[0]
            self.pos_y = pos[1]
            self.text = text
            self.font = font
            self.font_size = font_size
            self.font_scale = font_scale
            self.font_thickness = font_thickness
            self.color = color
            self.t = t
            text_size = getTextSize(self.text, self.font, self.font_size, self.font_scale, self.font_thickness)
            self.text_size = text_size
            self.type = type
            self.alpha = alpha

        def update_text_size(self):
            self.text_size = getTextSize(self.text, self.font, self.font_size, self.font_scale, self.font_thickness)

        def set_random_pos(self, x_range, y_range, textlabels=None, level_dict=None, min_distance=50, max_x=None, max_y=None):
            newx_range = x_range.copy()
            newy_range = y_range.copy()

            # Ensure text position stays within frame boundaries
            if max_x is not None and max_y is not None:
                # if newx_range[0] - self.text_size[0] < 0:
                #     newx_range[0] = self.text_size[0]
                if newx_range[1] + self.text_size[0] > max_x:
                    newx_range[1] = max_x - self.text_size[0]
                if newy_range[0] - self.text_size[1] < 0:
                    newy_range[0] = newy_range[0] + self.text_size[1]
                # if newy_range[1] + self.text_size[1] > max_y:
                #     newy_range[1] = max_y - self.text_size[1]*2
            
            self.pos_x, self.pos_y = random_position(newx_range, newy_range, text_size=self.text_size, textlabels=textlabels, level=level_dict[self.text] if level_dict is not None else None, min_distance=min_distance)
            return(self)

        def copy(self):
            return TextLabel(self.text, self.t, (self.pos_x, self.pos_y), self.type, self.font, self.font_size, self.font_scale, self.font_thickness, self.color, self.alpha)

    class FrameDisplay:
        """
        Class to store text labels and their properties.
        Parameters
        ----------
        labels : list
            The list of labels to display.
        scores : list
            The list of scores for each label.
        t : float
            The time at which the text label was created.
        level_dict : dict
            A dictionary containing the level of each label.
        type_dict : dict
            A dictionary containing the type of each label.
        font : int
            The font to use for the text label.
        font_thickness : int
            The font thickness to use for the text label.
        height : int
            The height of the frame.
        width : int
            The width of the frame.
        min_distance : int
            The minimum distance between text labels.
        font_size : int
            The font size to use for the text label.
        """
        def __init__(self, labels, scores, t, level_dict, type_dict, font=cv2.FONT_HERSHEY_COMPLEX, font_thickness=2, height=720, width=1280, min_distance=50, font_size=32):
            self.labels = labels
            self.scores = scores
            self.t = t
            self.height = height
            self.width = width
            self.font = font
            self.font_thickness = font_thickness
            self.level_dict = level_dict
            self.type_dict = type_dict
            self.min_distance = min_distance
            self.space = 0.20  # space in % from frame border
            self.x_range = [int(self.width * self.space), int(self.width * (1 - self.space))]
            self.y_range = [int(height * self.space), int(height * (1 - self.space))]
            self.font_size = font_size
            self.textlabels = [TextLabel(label, t=0, 
                                font_scale = self._font_scale_formula(score), font=self.font, 
                                font_thickness=self.font_thickness).set_random_pos(self.x_range, self.y_range, None, self.level_dict, 
                                self.min_distance, self.width, self.height)
                            for label, score in zip(labels, scores)]
            self.textlabels = [label for label in self.textlabels]

            self.fadingtextlabels = []
            
        def update(self, t, labels=None, scores=None):
            self.fadingtextlabels = self.fadingtextlabels + [textlabel for textlabel in self.textlabels if t - textlabel.t > 1]
            self.textlabels = [textlabel for textlabel in self.textlabels if t - textlabel.t <= 1]

            if labels is not None and scores is not None:
                for (label, score) in zip(labels, scores):
                    ############
                    #### DEALING WITH TEXT LABELS

                    # if label is a vanishing one
                    if self.type_dict[label] is None or self.type_dict[label] == "v":
                        if label not in [textlabel.text for textlabel in self.textlabels]:
                            newlabel = TextLabel(label, t=t, type=self.type_dict[label], 
                                                font_scale=self._font_scale_formula(scores[labels.index(label)]), 
                                                font=self.font, font_thickness=self.font_thickness)
                            newlabel.set_random_pos(self.x_range, self.y_range, self.textlabels, self.level_dict, self.min_distance, self.width, self.height)
                            self.textlabels.append(newlabel)
                        else:
                            textlabel = self.textlabels[[textlabel.text for textlabel in self.textlabels].index(label)]
                            textlabel.t = t
                            old_text_size = textlabel.text_size
                            textlabel.font_scale = self._font_scale_formula(scores[labels.index(label)])
                            textlabel.update_text_size()

                            textlabel.pos_x = round(textlabel.pos_x - (textlabel.text_size[0] - old_text_size[0])/2)
                            textlabel.pos_y = round(textlabel.pos_y + (textlabel.text_size[1] - old_text_size[1])/2)

                    # if label is a repeating one
                    elif self.type_dict[label] == "r":
                        repeat_threshold = 0.2
                        identical_labels = [textlabel for textlabel in self.textlabels if textlabel.text == label]
                        max_t = max([textlabel.t for textlabel in identical_labels]) if len(identical_labels) > 0 else -1
                        if t - max_t > repeat_threshold or max_t == -1:
                            if len(identical_labels) == 0:
                                x_range = self.x_range
                                y_range = self.y_range
                            else:
                                mean_x = np.mean([textlabel.pos_x for textlabel in identical_labels])
                                mean_y = np.mean([textlabel.pos_y for textlabel in identical_labels])
                                x_range = [int(mean_x - 100), int(mean_x + 100)]
                                y_range = [int(mean_y - 100), int(mean_y + 100)]
                            newlabel = TextLabel(label, t=t, type=self.type_dict[label], 
                                                font_scale=self._font_scale_formula(scores[labels.index(label)]), 
                                                font=self.font, font_thickness=self.font_thickness)
                            newlabel.set_random_pos(x_range, y_range, self.textlabels, self.level_dict, self.min_distance, self.width, self.height)
                            self.textlabels.append(newlabel)
                            # self.textlabels.append(TextLabel(random_position(x_range, y_range, level=self.level_dict[label], min_distance=self.min_distance),
                            #     label, t=t, type=self.type_dict[label], font_scale=self._font_scale_formula(scores[labels.index(label)]), font=self.font, font_thickness=self.font_thickness))
                            self.textlabels[-1].t = t
                            self.textlabels[-1].font_scale = self._font_scale_repeating_formula(scores[labels.index(label)])

            #######################
            ####### DEALING WITH FADING TEXT LABELS
            for textlabel in self.fadingtextlabels:
                textlabel.alpha = 1 / (t - textlabel.t)**1.7
            self.fadingtextlabels = [textlabel for textlabel in self.fadingtextlabels if textlabel.alpha > 0.01]

            self._fix_out_of_scope()

        def _font_scale_formula(self, score):
            return 0.5 + 2 * score

        def _font_scale_repeating_formula(self, score):
            return 0.5 + 1 * score

        def _fix_out_of_scope(self):
            # Ensure text position stays within frame boundaries

            for textlabel in self.textlabels+self.fadingtextlabels:
                if textlabel.pos_x + textlabel.text_size[0] > self.width:
                    textlabel.pos_x = self.width - textlabel.text_size[0]
                if textlabel.pos_y - textlabel.text_size[1] < 0:
                    textlabel.pos_y = textlabel.text_size[1]

    def make_frame(t):
        global scores_at_time_t
        global frame_display
        global frame_display_bg

        if t <= (len(audio)/sr)-flen:
            t_p = t
        else:
            t_p = (len(audio)/sr)-flen

        # Calculate elapsed time in minutes and seconds
        if label_datetimes != {}:
            time_float = make_timeline_frame(t, label_datetimes)
            time_str = convert_float_to_time(time_float)
            alpha = adjust_alpha(time_float)
        else:
            alpha = 0.8

        # Load background image
        height = 720
        width = 1280
        background = Image.open('./summaries_audiovisual/landscape.jpg').convert("RGBA")
        background = background.resize((width, height))  # Resize to match frame size

        # Apply Gaussian blur to the background image
        blurred_background = background.filter(ImageFilter.GaussianBlur(10))  # Adjust blur radius for intensity

        # Create a blank frame (black background)
        frame = Image.new('RGBA', (width, height), (0, 0, 0, 255))

        # Blend blurred background image with frame
        blended = Image.blend(frame, blurred_background, 1 - alpha)

        # Assuming labels is a list of lists where each sublist contains labels for each time t
        labels_at_time_t = labels[int(t_p * frame_rate)]
        scores_at_time_t = scores[int(t_p * frame_rate)]

        labels_at_time_t_bg = labels_bg[int(t_p * frame_rate)]
        scores_at_time_t_bg = scores_bg[int(t_p * frame_rate)]

        min_distance = ((width + height) / 2) / 30
        font_path = "./fonts/Girls Have Many Secrets.ttf"
        font_size = 44
        font_thickness = 2

        if t == 0.0:
            frame_display = FrameDisplay(labels_at_time_t, scores_at_time_t, t, level_dict, type_dict,
                                        font=font_path, font_thickness=font_thickness, height=height,
                                        width=width, min_distance=min_distance, font_size=font_size)
            frame_display_bg = FrameDisplay(labels_at_time_t_bg, scores_at_time_t_bg, t, level_dict, type_dict,
                                            font=font_path, font_thickness=font_thickness, height=height,
                                            width=width, min_distance=min_distance, font_size=font_size)
        else:
            frame_display.update(t, labels_at_time_t, scores_at_time_t)
            frame_display_bg.update(t, labels_at_time_t_bg, scores_at_time_t_bg)

        # Add labels at random positions
        for label in frame_display.textlabels + frame_display.fadingtextlabels + frame_display_bg.textlabels + frame_display_bg.fadingtextlabels:

            txt = Image.new('RGBA', frame.size, (0, 0, 0, 0))
            font_label = ImageFont.truetype(font_path, round(font_size*label.font_scale))
            d = ImageDraw.Draw(txt)
            stroke = int(255*label.alpha)
            d.text((label.pos_x, label.pos_y), f"{label.text}", fill=label.color + (int(255*label.alpha),), font=font_label, stroke_width=3, stroke_fill=(0, 0, 0, stroke))
            blended = Image.alpha_composite(blended, txt)

        if label_datetimes != {}:
            # Draw the time on a clock face
            clock_radius = 50
            clock_center = (int(blended.width * 0.03 + clock_radius), int(blended.height * 0.12))
            clock_bg_color = (0, 0, 0, 255)
            clock_hand_color = (255, 255, 255, 255)

            txt = Image.new('RGBA', frame.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(txt)

            # Draw clock face
            d.ellipse((clock_center[0] - clock_radius, clock_center[1] - clock_radius,
                    clock_center[0] + clock_radius, clock_center[1] + clock_radius), fill=clock_bg_color)

            # Calculate clock hand position
            angle = (time_float%24 / 24.0) * 360.0  # Convert time to angle
            angle_rad = math.radians(angle)
            hand_length = clock_radius * 0.8
            hand_x = clock_center[0] + hand_length * math.cos(angle_rad - math.pi/2)
            hand_y = clock_center[1] + hand_length * math.sin(angle_rad - math.pi/2)

            font_clock_path = "./fonts/00TT.TTF"
            font_clock_size = 20
            clock_number_color = (255, 255, 255, 255)
            font_clock = ImageFont.truetype(font_clock_path, font_clock_size)

            # Draw clock numbers
            number_positions = {
                "0": (clock_center[0], clock_center[1] - clock_radius * 1.3),
                "6": (clock_center[0] + clock_radius * 1.3, clock_center[1]),
                "12": (clock_center[0], clock_center[1] + clock_radius * 1.3),
                "18": (clock_center[0] - clock_radius * 1.3, clock_center[1]),
            }
            for number, position in number_positions.items():
                d.text(position, number, fill=clock_number_color, font=font_clock, anchor="mm")

            # Draw clock hand
            d.line((clock_center, (hand_x, hand_y)), fill=clock_hand_color, width=4)
            blended = Image.alpha_composite(blended, txt)

        # DO NOT DELETE: version with resizing of text image instead of font: issue with positioning
        # #here the maximum font size is font_size. The "normal" font_size is 32.
        # size_match = min(label.font_scale / 2, 1)
        # # Reduce the size of the text image
        # txt = txt.resize((int(txt.width * size_match), int(txt.height * size_match)))
        
        # pos_x = label.pos_x * size_match
        # pos_y = label.pos_y * size_match

        # delta_x = label.pos_x - pos_x
        # delta_y = label.pos_y - pos_y

        # # Fill the remaining text image with blank
        # blank = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        # blank.paste(txt, (int(delta_x), int(delta_y)))
        # blended = Image.alpha_composite(blended, blank)

        return np.array(blended)


    ###############
    ############
    ## TEMP FILES

    # Define parameters
    temp_output_video_path = "./summaries_audiovisual/video/.temp/.temp.mp4"

    # Check if the directory exists, if not, create it
    directory = os.path.dirname(temp_output_video_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if temp.mp4 exists, if yes, delete it
    if os.path.exists(temp_output_video_path):
        os.remove(temp_output_video_path)

    temp_audio_path = "./summaries_audiovisual/video/.temp/.temp.wav"
    # Normalize the audio using ffmpeg

    # Check if the directory exists, if not, create it
    directory = os.path.dirname(temp_audio_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if temp.wav exists, if yes, delete it
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    waveform = librosa.util.normalize(waveform)

    sf.write(temp_audio_path, waveform, sr)

    ###############
    ############

    duration = len(audio)/sr    # Duration of the video in seconds

    writer = imageio.get_writer(temp_output_video_path, format='ffmpeg', mode='I', fps=int(frame_rate))

    # Generate frames and add them to the video
    for t in range(int(duration * frame_rate)):
        frame = make_frame(t / frame_rate)  # Generate frame for current time
        writer.append_data(frame)
        print(f"Progress: {t+1}/{int(duration * frame_rate)}", end='\r')

    # Close the writer
    writer.close()

    # Use ffmpeg to combine video and audio
    command = [
        'ffmpeg', '-y', '-i', temp_output_video_path, '-i', temp_audio_path, '-c:v', 'copy', '-c:a', 'aac',
        '-strict', 'experimental', '-map', '0:v:0', '-map', '1:a:0', output_video_path
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