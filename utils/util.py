import os
import torch
import yaml
import yamlloader
import numpy as np
import matplotlib.pyplot as plt
import re
import send2trash
import math
import pickle 
import utils.bands_transform as bt
from prettytable import PrettyTable
import matplotlib as mpl
import librosa
import soundfile as sf
from torchaudio.transforms import Fade
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS, CONVTASNET_BASE_LIBRI2MIX
import random 
from itertools import groupby
import torchaudio
from scipy.signal import stft, istft
import scipy.interpolate
from torchaudio.transforms import Resample

def chunks(lst, n):
    """
    Yield successive n-sized chunks from a list.

    Args:
        lst (list): The input list.
        n (int): The size of each chunk.

    Yields:
        list: A chunk of size n from the input list.

    Examples:
        >>> lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> for chunk in chunks(lst, 3):
        ...     print(chunk)
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class RandomHopAudioChunks():
    def __init__(self, winmin, winmax, hop_perc=0.90, fade=True):
        # window size limits
        self.winmin = winmin
        self.winmax = winmax
        # whether to apply a fade in and fade out on each chunk or not
        self.fade = fade
        self.hop_perc = hop_perc
    
    def chunks_with_hop(self, lst):

        L = []
        idx = 0

        # Generate chunks with random windows
        while idx < len(lst):
            # Choose a random window size between winmin and winmax
            window_size = random.randint(self.winmin, self.winmax)
            # Calculate the chunk size (0.95 of the window size)
            chunk_size = int(self.hop_perc * window_size)
            end_idx = idx + window_size

            if end_idx > len(lst):  # Handle the last chunk if it exceeds the list length
                end_idx = len(lst)
            
            L.append(lst[idx:end_idx])
            
            # Move the index to the next position for chunking
            idx += chunk_size
        
        return L

    def concat_with_hop(self, L):
        # Initialize the output list with the correct size
        total_length = sum(len(chunk) for chunk in L) - sum(int((1-self.hop_perc) * len(chunk)) for chunk in L[:-1])
        lst = np.zeros(shape=total_length)

        # Keep track of the position in the output list
        current_position = 0

        for i, chunk in enumerate(L):
            chunk_length = len(chunk)
            # Calculate diffhop as 5% of the current chunk size (overlap)
            diffhop = int((1-self.hop_perc) * chunk_length)

            # If fade is enabled, calculate the fade in/out windows
            if self.fade:
                bef = np.linspace(0, 1, diffhop)
                aft = np.linspace(1, 0, diffhop)
                mid = np.ones(chunk_length - diffhop)
                pond_g = np.concatenate((bef, mid))
                pond_d = np.concatenate((mid, aft))
            else:
                pond_g = np.ones(chunk_length)
                pond_d = np.ones(chunk_length)

            # If it's the first chunk, simply copy it to the output list
            if i == 0:
                lst[current_position:current_position + chunk_length] = pond_d * chunk
                current_position += chunk_length
            else:
                # Calculate the starting position in the output list considering overlap
                overlap_position = current_position - diffhop

                # Apply the overlap
                if overlap_position + chunk_length > len(lst):
                    break

                lst[overlap_position:overlap_position+chunk_length] += pond_g * pond_d * chunk

                # Update the position for the next chunk
                current_position += (chunk_length - diffhop)

        return lst

class FullRandomHopAudioChunksShuffle():
    def __init__(self, winmin, winmax, hopmin_perc=0.10, hopmax_perc=0.90, fade=True):
        # window size limits
        self.winmin = winmin
        self.winmax = winmax
        # hop size limits as percentages of the window size
        self.hopmin_perc = hopmin_perc
        self.hopmax_perc = hopmax_perc
        # whether to apply a fade in and fade out on each chunk or not
        self.fade = fade
        
        # To store window lengths and hop lengths
        self.wins = []
        self.diffhops = []
        self.pos_l = []
        self.pos_r = []

    def chunks_with_hop(self, lst, mixwin=True, chunk_perc=0.1):

        L = []
        idx = 0
        while idx < len(lst):
            # Choose a random window size between winmin and winmax
            window_size = random.randint(self.winmin, self.winmax)
            self.pos_l.append(idx)
            self.pos_r.append(idx+window_size)
            self.wins.append(window_size)  # Store window size
            L.append(lst[self.pos_l[-1]:self.pos_r[-1]])
            idx = idx + window_size
        L[-1] = lst[self.pos_l[-1]:len(lst)]
        self.wins[-1] = len(L[-1])
        self.pos_r[-1] = len(lst)

        # mytest = [len(chunk)-win for _, (chunk, win) in enumerate(zip(L, self.wins))]
        # mytest = [0 if t >= 0 else -1 for t in mytest]
        # print('INPUT')
        # print(mytest)
        # print(len(L))
        # print(len(self.wins))

        # Shuffle self.wins, self.pos_l, and self.pos_r together
        # if mixwin:
        #     combined = list(zip(L, self.wins, self.pos_l, self.pos_r))
        #     random.shuffle(combined)
        #     L, self.wins, self.pos_l, self.pos_r = zip(*combined)
        #     L = list(L)
        #     self.wins = list(self.wins)
        #     self.pos_l = list(self.pos_l)
        #     self.pos_r = list(self.pos_r)

        # print('TTTTTTTTTTTTTTT')
        # print(len(L))
        # print(np.sum(len(L[i]) for i in range(len(L))))

        if mixwin:
            combined = list(zip(L, self.wins, self.pos_l, self.pos_r))
            chunk_size = max(1, int((len(lst) * chunk_perc) // (len(lst) / len(L))))   # Number of chunks per group to shuffle
            shuffled_combined = []

            for i in range(0, len(combined), chunk_size):
                group = combined[i:i + chunk_size]  # Get the group of chunks
                random.shuffle(group)  # Shuffle only this group
                shuffled_combined.extend(group)  # Add the shuffled group to the result

                # test1 = [len(chunk)-win for chunk, win, _, _ in group]
                # test2 = [len(chunk) for chunk, win, _, _ in group]
                # test3 = [win for chunk, win, _, _ in group]
                # print('PPPPPPPP')
                # print(test1)
                # print(test2)
                # print(test3)

            L, self.wins, self.pos_l, self.pos_r = zip(*shuffled_combined)
            L = list(L)
            self.wins = list(self.wins)
            self.pos_l = list(self.pos_l)
            self.pos_r = list(self.pos_r)

        self.newpos_l = list(self.pos_l)
        self.newpos_r = list(self.pos_r)

        # print('LLLLLLLLLLLL')
        # print([len(chunk)-win for _, (chunk, win) in enumerate(zip(L, self.wins))])

        for i in range(1, len(L)):
            minwinsize = min([self.wins[i], self.wins[i-1]])
            hop_perc = random.uniform(self.hopmin_perc, self.hopmax_perc)
            diffhop = int((1-hop_perc) * minwinsize)
            self.diffhops.append(diffhop)
            diffhop_l = diffhop // 2
            diffhop_r = diffhop - diffhop_l

            self.newpos_l[i] = self.newpos_l[i] - diffhop_l
            self.newpos_r[i-1] = self.newpos_r[i-1] + diffhop_r

        for i, (pos_l, pos_r, newpos_l, newpos_r) in enumerate(zip(self.pos_l, self.pos_r, self.newpos_l, self.newpos_r)):
            if newpos_l < 0:
                L[i] = np.concatenate((L[i][pos_l:2*pos_l-newpos_l][::-1],L[i],lst[pos_r:newpos_r]))

            elif newpos_r > len(lst):
                L[i] = np.concatenate((lst[newpos_l:pos_l], L[i], lst[2*pos_r-newpos_r:pos_r][::-1]))

            else:
                L[i] = np.concatenate((lst[newpos_l:pos_l],L[i],lst[pos_r:newpos_r]))
            self.wins[i] = len(L[i])

            # if len(L[i]) == 6815:
            #     print('XXXXXXXXXXXXXXXXXXXXXXXXX')
            #     print(newpos_l-pos_l)

        # mytest = [len(chunk)-diff for _, (chunk, diff) in enumerate(zip(L, self.diffhops))]
        # mytest = [0 if t > 0 else -1 for t in mytest]
        # print('INPUT')
        # print(mytest)

        return L

    def concat_with_hop(self, L):
        # Initialize the output list with the correct size
        total_length = sum(len(chunk) for chunk in L) - sum(diffhop for diffhop in self.diffhops)
        # total_length = sum(win for win in self.wins) - sum(diffhop for diffhop in self.diffhops)

        # print('OOOOOOOOOO')
        # print([len(chunk) for chunk in L])

        lst = np.zeros(shape=total_length)

        # Keep track of the position in the output list
        current_position = 0

        for i, chunk in enumerate(L):
            chunk_length = len(chunk)

            # Calculate diffhop as 5% of the current chunk size (overlap)
            if (i == (len(L)-1)):
                diffhop = 0
            else:
                diffhop = self.diffhops[i]

            # If fade is enabled, calculate the fade in/out windows
            if self.fade:
                bef = np.linspace(0, 1, diffhop)
                aft = np.linspace(1, 0, diffhop)
                mid = np.ones(chunk_length - diffhop)
                pond_g = np.concatenate((bef, mid))
                pond_d = np.concatenate((mid, aft))
            else:
                pond_g = np.ones(chunk_length)
                pond_d = np.ones(chunk_length)

            # If it's the first chunk, simply copy it to the output list
            if i == 0:
                lst[current_position:current_position + chunk_length] = pond_d * chunk
                current_position += chunk_length
            else:
                # Calculate the starting position in the output list considering overlap
                overlap_position = current_position - diffhop

                # Apply the overlap
                if overlap_position + chunk_length > len(lst):
                    break

                if overlap_position < 0:
                    break

                lst[overlap_position:overlap_position+chunk_length] += pond_g * pond_d * chunk

                # Update the position for the next chunk
                current_position += (chunk_length - diffhop)
        
        return lst

class FullRandomHopAudioChunks():
    def __init__(self, winmin, winmax, hopmin_perc=0.10, hopmax_perc=0.90, fade=True):
        # window size limits
        self.winmin = winmin
        self.winmax = winmax
        # hop size limits as percentages of the window size
        self.hopmin_perc = hopmin_perc
        self.hopmax_perc = hopmax_perc
        # whether to apply a fade in and fade out on each chunk or not
        self.fade = fade
        
        # To store window lengths and hop lengths
        self.wins = []
        self.diffhops = []

    def chunks_with_hop(self, lst):
        L = []

        window_size = random.randint(self.winmin, self.winmax)
        L.append(lst[0:window_size])
        self.wins.append(window_size)  # Store window size

        idx = window_size

        # Generate chunks with random windows and hops
        while idx < len(lst):
            # Choose a random window size between winmin and winmax
            window_size = random.randint(self.winmin, self.winmax)
            minwinsize = min([window_size, self.wins[-1]])
            hop_perc = random.uniform(self.hopmin_perc, self.hopmax_perc)
            # hop_size = int(hop_perc * minwinsize)
            diffhop = int((1-hop_perc) * minwinsize)

            if idx + window_size - diffhop > len(lst):
                diffhop = int((1-hop_perc) * self.wins[-1])
                window_size = len(lst) - idx + diffhop  # Adjust window size to include the remaining data

            self.diffhops.append(diffhop)
            self.wins.append(window_size)  # Store window size

            idx -= diffhop
            L.append(lst[idx:idx+window_size])
            idx += window_size

        # # Generate chunks with random windows and hops
        # while idx < len(lst):
        #     # Choose a random window size between winmin and winmax
        #     window_size = random.randint(self.winmin, self.winmax)
        #     end_idx = idx + window_size

        #     if end_idx > len(lst):  # Handle the last chunk if it exceeds the list length
        #         end_idx = len(lst)
        #         window_size = len(lst) - idx  # Adjust window size to include the remaining data
        #         self.wins.append(window_size)  # Store window size
        #         L.append(lst[idx:end_idx])
        #         # Choose a random hop size as a percentage of the window size
        #         hop_perc = random.uniform(self.hopmin_perc, self.hopmax_perc)
        #         hop_size = int(hop_perc * window_size)
        #         # self.hops.append(hop_size)  # Store hop size
        #         self.diffhops.append(int((1-hop_perc) * window_size))
        #         break # Stop the loop if the last chunk is smaller than the window size

        #     self.wins.append(window_size)  # Store window size

        #     if idx != 0:
        #         # Choose a random hop size as a percentage of the window size
        #         hop_perc = random.uniform(self.hopmin_perc, self.hopmax_perc)
        #         hop_size = int(hop_perc * window_size)
        #         self.diffhops.append(int((1-hop_perc) * window_size))

        #     L.append(lst[idx:end_idx])

        #     # Move the index to the next position for chunking
        #     if idx != 0:
        #         idx += window_size - hop_size
        #     else:
        #         idx = window_size

        return L

    def concat_with_hop(self, L):
        # Initialize the output list with the correct size
        total_length = sum(len(chunk) for chunk in L) - sum(diffhop for diffhop in self.diffhops)
        # total_length = sum(win for win in self.wins) - sum(diffhop for diffhop in self.diffhops)

        lst = np.zeros(shape=total_length)

        # Keep track of the position in the output list
        current_position = 0

        for i, chunk in enumerate(L):
            chunk_length = len(chunk)
            # Calculate diffhop as 5% of the current chunk size (overlap)
            if (i == (len(L)-1)):
                diffhop = 0
            else:
                diffhop = self.diffhops[i]

            # If fade is enabled, calculate the fade in/out windows
            if self.fade:
                bef = np.linspace(0, 1, diffhop)
                aft = np.linspace(1, 0, diffhop)
                mid = np.ones(chunk_length - diffhop)
                pond_g = np.concatenate((bef, mid))
                pond_d = np.concatenate((mid, aft))
            else:
                pond_g = np.ones(chunk_length)
                pond_d = np.ones(chunk_length)

            # If it's the first chunk, simply copy it to the output list
            if i == 0:
                lst[current_position:current_position + chunk_length] = pond_d * chunk
                current_position += chunk_length
            else:
                # Calculate the starting position in the output list considering overlap
                overlap_position = current_position - diffhop

                # Apply the overlap
                if overlap_position + chunk_length > len(lst):
                    break

                if overlap_position < 0:
                    break

                lst[overlap_position:overlap_position+chunk_length] += pond_g * pond_d * chunk

                # Update the position for the next chunk
                current_position += (chunk_length - diffhop)

        return lst


# class FullRandomHopAudioChunks():
#     def __init__(self, winmin, winmax, hopmin_perc=0.10, hopmax_perc=0.90, fade=True):
#         # window size limits
#         self.winmin = winmin
#         self.winmax = winmax
#         # hop size limits as percentages of the window size
#         self.hopmin_perc = hopmin_perc
#         self.hopmax_perc = hopmax_perc
#         # whether to apply a fade in and fade out on each chunk or not
#         self.fade = fade
        
#         # To store window lengths and hop lengths
#         self.wins = []
#         self.hops = []

#     def chunks_with_hop(self, lst):
#         L = []
#         idx = 0

#         # Generate chunks with random windows and hops
#         while idx < len(lst):
#             # Choose a random window size between winmin and winmax
#             window_size = random.randint(self.winmin, self.winmax)
#             end_idx = idx + window_size

#             if end_idx > len(lst):  # Handle the last chunk if it exceeds the list length
#                 end_idx = len(lst)
#                 window_size = len(lst) - idx  # Adjust window size to include the remaining data
#                 self.wins.append(window_size)  # Store window size
#                 L.append(lst[idx:end_idx])
#                 break # Stop the loop if the last chunk is smaller than the window size

#             self.wins.append(window_size)  # Store window size

#             # Choose a random hop size as a percentage of the window size
#             hop_perc = random.uniform(self.hopmin_perc, self.hopmax_perc)
#             hop_size = int(hop_perc * window_size)
#             self.hops.append(hop_perc)  # Store hop size

#             L.append(lst[idx:end_idx])

#             # Move the index to the next position for chunking
#             idx += hop_size

#         return L

#     def concat_with_hop(self, L):
#         # Initialize the output list with the correct size
#         total_length = sum(len(chunk) for chunk in L) - sum(int((1-self.hops[i]) * len(chunk)) for i, chunk in enumerate(L[:-1]))
#         lst = np.zeros(shape=total_length)

#         # Keep track of the position in the output list
#         current_position = 0

#         for i, chunk in enumerate(L):
#             chunk_length = len(chunk)
#             # Calculate diffhop as 5% of the current chunk size (overlap)
#             diffhop = int((1-self.hops[i]) * chunk_length)

#             # If fade is enabled, calculate the fade in/out windows
#             if self.fade:
#                 bef = np.linspace(0, 1, diffhop)
#                 aft = np.linspace(1, 0, diffhop)
#                 mid = np.ones(chunk_length - diffhop)
#                 pond_g = np.concatenate((bef, mid))
#                 pond_d = np.concatenate((mid, aft))
#             else:
#                 pond_g = np.ones(chunk_length)
#                 pond_d = np.ones(chunk_length)

#             # If it's the first chunk, simply copy it to the output list
#             if i == 0:
#                 lst[current_position:current_position + chunk_length] = pond_d * chunk
#                 current_position += chunk_length
#             else:
#                 # Calculate the starting position in the output list considering overlap
#                 overlap_position = current_position - diffhop

#                 # Apply the overlap
#                 if overlap_position + chunk_length > len(lst):
#                     break

#                 if overlap_position < 0:
#                     break

#                 lst[overlap_position:overlap_position+chunk_length] += pond_g * pond_d * chunk

#                 # Update the position for the next chunk
#                 current_position += (chunk_length - diffhop)

#         return lst
    
    # def concat_with_hop(self, L):
    #     # Initialize the output list with the correct size
    #     total_length = sum(len(chunk) for chunk in L) - sum(self.hops[:-1]) + len(L[-1])
    #     lst = np.zeros(shape=total_length)

    #     # Keep track of the position in the output list
    #     current_position = 0

    #     for i, chunk in enumerate(L):
    #         chunk_length = len(chunk)
    #         hop_size = self.hops[i]
    #         window_size = self.wins[i]

    #         if chunk_length < window_size:
    #             print(f'WARNING: Chunk length {chunk_length} is smaller than the window size {window_size}.')

    #         # If fade is enabled, calculate the fade in/out windows
    #         if self.fade:
    #             fade_in = np.linspace(0, 1, hop_size)
    #             fade_out = np.linspace(1, 0, hop_size)
    #             mid = np.ones(chunk_length - hop_size)
    #             fade_g = np.concatenate((fade_in, mid))
    #             fade_d = np.concatenate((mid, fade_out))
    #         else:
    #             fade_g = np.ones(chunk_length)
    #             fade_d = np.ones(chunk_length)

    #         # If it's the first chunk, simply copy it to the output list
    #         if i == 0:
    #             lst[current_position:current_position + chunk_length] = fade_d * chunk
    #             current_position += chunk_length
    #         else:
    #             # Calculate the starting position in the output list considering overlap
    #             overlap_position = current_position + window_size - hop_size

    #             # Apply the overlap
    #             if overlap_position + chunk_length > len(lst):
    #                 break

    #             if overlap_position < 0:
    #                 break
                
    #             # print('LLLLLLLLLLL')
    #             # print(overlap_position)
    #             # print(chunk_length)
    #             # print(hop_size)
    #             # print(current_position)
    #             # print(fade_g.shape, fade_d.shape, chunk.shape, lst[overlap_position:overlap_position + chunk_length].shape)
    #             lst[overlap_position:overlap_position + chunk_length] += fade_g * fade_d * chunk

    #             # Update the position for the next chunk
    #             # current_position += (chunk_length - hop_size)
    #             current_position += hop_size

    #     return lst

# class RandomHopAudioChunks():
#     def __init__(self, n, winmin, winmax, fade=True):
#         #number of elements in each chunk
#         self.n = n
#         #size of hop
#         self.hopmin = hopmin
#         self.hopmax = hopmax
#         self.diffhop = n - hop

#         #whether to apply a fade in and fade out on each chunk or not
#         self.fade = fade
    
#     def chunks_with_hop(self, lst):
#         if self.n != self.diffhop:
#             L = []
#             L.append(lst[0:self.n])
#             idx = 0

#             if self.n == self.diffhop:
#                 step = self.n
#             else:
#                 step = self.n - self.diffhop

#             for i in range(self.n, len(lst) - self.n + self.diffhop, step):
#                 L.append(lst[(i - self.diffhop):(i + self.n - self.diffhop)])
#                 idx = i
#             if idx + 2 * (self.n - self.diffhop) == len(lst):
#                 L.append(lst[len(lst) - self.n:len(lst)])
#         else:
#             L = []
#             step = self.n
#             for i in range(0, len(lst), step):
#                 to_add = lst[i:i + step]
#                 if len(to_add) == step:
#                     L.append(to_add)

#         return np.array(L)

#     def concat_with_hop(self, L):
#         lst = np.zeros(shape=L.shape[1] * L.shape[0] - (L.shape[0] - 1) * self.diffhop)
#         if self.fade:
#             bef = np.linspace(0, 1, self.diffhop)
#             aft = np.linspace(1, 0, self.diffhop)
#             mid = np.ones(L.shape[1] - self.diffhop)
#             pond_g = np.concatenate((bef, mid))
#             pond_d = np.concatenate((mid, aft))
#         else:
#             pond_g = np.ones(L.shape[1])
#             pond_d = np.ones(L.shape[1])

#         lst[0:L.shape[1]] = pond_d * L[0, :]
#         for i in range(1, L.shape[0]):
#             if i != L.shape[0] - 1:
#                 lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * pond_d * L[i, :]
#             else:
#                 lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * L[i, :]

#         return lst

class AudioChunks():
    def __init__(self, n, hop, fade=True):
        #number of elements in each chunk
        self.n = n
        #size of hop
        self.hop = hop
        self.diffhop = n - hop

        #whether to apply a fade in and fade out on each chunk or not
        self.fade = fade

    def calculate_num_chunks(self, wavesize):
        num_chunks = 1
        idx = 0
        audio_truncated=False
        
        if self.n == self.diffhop:
            step = self.n
        else:
            step = self.n-self.diffhop

        for i in range(self.n, wavesize-self.n+self.diffhop, step):
            num_chunks += 1
            idx = i

        if idx+2*(self.n-self.diffhop) == wavesize:
            num_chunks += 1
        else:
            audio_truncated=True

        if self.n == self.diffhop:
            if self.n*num_chunks == wavesize:
                audio_truncated=False
            else:
                audio_truncated=True
            
        return(num_chunks, audio_truncated)
    
    def chunks_with_hop(self, lst):
        if isinstance(lst, np.ndarray):
            return self._chunks_with_hop_np(lst)
        elif isinstance(lst, torch.Tensor):
            return self._chunks_with_hop_torch(lst)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _chunks_with_hop_np(self, lst):
        if self.n != self.diffhop:
            L = []
            L.append(lst[0:self.n])
            idx = 0

            if self.n == self.diffhop:
                step = self.n
            else:
                step = self.n - self.diffhop

            for i in range(self.n, len(lst) - self.n + self.diffhop, step):
                L.append(lst[(i - self.diffhop):(i + self.n - self.diffhop)])
                idx = i
            if idx + 2 * (self.n - self.diffhop) == len(lst):
                L.append(lst[len(lst) - self.n:len(lst)])
        else:
            L = []
            step = self.n
            for i in range(0, len(lst), step):
                to_add = lst[i:i + step]
                if len(to_add) == step:
                    L.append(to_add)

        return np.array(L)

    def _chunks_with_hop_torch(self, lst):
        if self.n != self.diffhop:
            L = []
            L.append(lst[0:self.n])
            idx = 0

            if self.n == self.diffhop:
                step = self.n
            else:
                step = self.n - self.diffhop

            for i in range(self.n, len(lst) - self.n + self.diffhop, step):
                L.append(lst[(i - self.diffhop):(i + self.n - self.diffhop)])
                idx = i
            if idx + 2 * (self.n - self.diffhop) == len(lst):
                L.append(lst[len(lst) - self.n:len(lst)])
        else:
            L = []
            step = self.n
            for i in range(0, len(lst), step):
                to_add = lst[i:i + step]
                if len(to_add) == step:
                    L.append(to_add)

        return torch.stack(L)

    def concat_with_hop(self, L):
        if isinstance(L, np.ndarray):
            return self._concat_with_hop_np(L)
        elif isinstance(L, torch.Tensor):
            return self._concat_with_hop_torch(L)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _concat_with_hop_np(self, L):
        lst = np.zeros(shape=L.shape[1] * L.shape[0] - (L.shape[0] - 1) * self.diffhop)
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[1] - self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))
        else:
            pond_g = np.ones(L.shape[1])
            pond_d = np.ones(L.shape[1])

        lst[0:L.shape[1]] = pond_d * L[0, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * pond_d * L[i, :]
            else:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * L[i, :]

        return lst

    def _concat_with_hop_torch(self, L):
        lst = torch.zeros(L.shape[1] * L.shape[0] - (L.shape[0] - 1) * self.diffhop)
        if self.fade:
            bef = torch.linspace(0, 1, self.diffhop)
            aft = torch.linspace(1, 0, self.diffhop)
            mid = torch.ones(L.shape[1] - self.diffhop)
            pond_g = torch.cat((bef, mid))
            pond_d = torch.cat((mid, aft))
        else:
            pond_g = torch.ones(L.shape[1])
            pond_d = torch.ones(L.shape[1])

        lst[0:L.shape[1]] = pond_d * L[0, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * pond_d * L[i, :]
            else:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * L[i, :]

        return lst

    def concat_spec_with_hop(self, L):
        if isinstance(L, np.ndarray):
            return self._concat_spec_with_hop_np(L)
        elif isinstance(L, torch.Tensor):
            return self._concat_spec_with_hop_torch(L)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _concat_spec_with_hop_np(self, L):
        lst = np.zeros(shape=(L.shape[1], L.shape[2] * L.shape[0] - (L.shape[0] - 1) * self.diffhop))
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[2] - self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))

            pond_g = np.tile(pond_g, (L.shape[1], 1))
            pond_d = np.tile(pond_d, (L.shape[1], 1))
        else:
            pond_g = np.ones((L.shape[1], L.shape[2]))
            pond_d = np.ones((L.shape[1], L.shape[2]))

        lst[:, 0:L.shape[2]] = pond_d * L[0, :, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * pond_d * L[i, :, :]
            else:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * L[i, :, :]

        return lst

    def _concat_spec_with_hop_torch(self, L):
        lst = torch.zeros((L.shape[1], L.shape[2] * L.shape[0] - (L.shape[0] - 1) * self.diffhop))
        if self.fade:
            bef = torch.linspace(0, 1, self.diffhop)
            aft = torch.linspace(1, 0, self.diffhop)
            mid = torch.ones(L.shape[2] - self.diffhop)
            pond_g = torch.cat((bef, mid))
            pond_d = torch.cat((mid, aft))

            pond_g = pond_g.repeat(L.shape[1], 1)
            pond_d = pond_d.repeat(L.shape[1], 1)
        else:
            pond_g = torch.ones((L.shape[1], L.shape[2]))
            pond_d = torch.ones((L.shape[1], L.shape[2]))

        lst[:, 0:L.shape[2]] = pond_d * L[0, :, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * pond_d * L[i, :, :]
            else:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * L[i, :, :]

        return lst


class SettingsLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(SettingsLoader, self).__init__(stream)
    
    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            #return yaml.load(f, YAMLLoader)
            return yaml.load(f, yamlloader)

SettingsLoader.add_constructor('!include', SettingsLoader.include)

def load_settings(file_path):
    with file_path.open('r') as f:
        return yaml.load(f, Loader=SettingsLoader)

#MT: added
def plot_spectro(x_m, fs, extlmax=None, title='title', vmin=None, vmax=None, save=False):
    if vmin==None:
        vmin = np.min(x_m)
    if vmax==None:
        vmax = np.max(x_m)
    exthmin = 1
    exthmax = len(x_m)
    extlmin = 0
    if extlmax==None:
        extlmax = len(x_m[0])

    plt.figure(figsize=(8, 5))
    plt.imshow(x_m, extent=[extlmin,extlmax,exthmin,exthmax], cmap='inferno',
               vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
    plt.colorbar()
    plt.title(title)
    if save:
        plt.savefig(f"./figures/{title}.png")
    plt.show()

#MT: added
class ChunkManager():
    def __init__(self, dataset_name, model_name, model_batch_path, batch_type_name, batch_lim=1000):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_batch_path = model_batch_path
        self.batch_type_name = batch_type_name
        self.current_batch_id = 0
        self.batch_lim = batch_lim
        self.total_folder_name = self.batch_type_name + '_' + self.model_name + '_' + self.dataset_name
        self.total_path = self.model_batch_path / self.total_folder_name
        if not os.path.exists(self.total_path):
            os.makedirs(self.total_path)
        else:
            print(f'WARNING: everything will be deleted in path: {self.total_path}')
            self._delete_everything_in_folder(self.total_path)
        
    def save_chunk(self, batch, forced=False):
        if len(batch) == 0:
            return(batch)
        if len(batch) >= self.batch_lim or forced == True:
            file_path = self.total_path / (self.total_folder_name + '_' + str(self.current_batch_id) + '.npy')
            np.save(file_path, batch)
            print(f'save made in: {file_path}')
            self.current_batch_id+=1
            return([])
        else:
            return(batch)
        
    def open_chunks(self):
        stacked_batch = np.array([])
        for root, dirs, files in os.walk(self.total_path):
            
            #sort files
            files_splitted = [re.split(r'[_.]', file) for file in files]
            file_indices = [int(file[-2]) for file in files_splitted]
            file_indices_sorted = file_indices.copy()
            file_indices_sorted.sort()
            file_new_indices = [file_indices.index(ind) for ind in file_indices_sorted]
            files_sorted = [files[i] for i in file_new_indices]
            

            for file in files_sorted:
                cur_batch = np.load(self.total_path / file, allow_pickle=True)
                stacked_batch = np.concatenate((stacked_batch, cur_batch))

        return(stacked_batch)
            
    def _delete_everything_in_folder(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    send2trash(file_path)
                    #shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def save_predictions(files, predictions, path, name):
    
    pred_dict = dict(zip(files, predictions))
    # define a dictionary with key value pairs
    #dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}
    
    with open(path / (name +'.pkl'), 'wb') as f:
        pickle.dump(pred_dict, f)
        
def load_predictions(path, name):
    
    with open(path / (name +'.pkl'), 'rb') as f:
        loaded_dict = pickle.load(f)
    
    return(loaded_dict)

def tukey_window(M, alpha=0.2):
    """Return a Tukey window, also known as a tapered cosine window, and an 
    energy correction value to make sure to preserve energy.
    Window and energy correction calculated according to:
    https://github.com/nicolas-f/noisesensor/blob/master/core/src/acoustic_indicators.c#L150

    Parameters
    ----------
    M : int
        Number of points in the output window. 
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.

    Returns
    -------
    window : ndarray
        The window, with the maximum value normalized to 1.
    energy_correction : float
        The energy_correction used to compensate the loss of energy due to
        the windowing
    """
    #nicolas' calculation
    index_begin_flat = int((alpha / 2) * M)
    index_end_flat = int(M - index_begin_flat)
    energy_correction = 0
    window = np.zeros(M)
    
    for i in range(index_begin_flat):
        window_value = (0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - alpha / 2))))
        energy_correction += window_value * window_value
        window[i]=window_value
    
    energy_correction += (index_end_flat - index_begin_flat) #window*window=1
    for i in range(index_begin_flat, index_end_flat):
        window[i] = 1
    
    for i in range(index_end_flat, M):
        window_value = (0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - 1 + alpha / 2))))
        energy_correction += window_value * window_value
        window[i] = window_value
    
    energy_correction = 1 / math.sqrt(energy_correction / M)
    
    return(window, energy_correction)

def get_transforms(sr=32000, flen=4096, hlen=4000, classifier='YamNet', device=torch.device("cpu"), tho_freq=True, tho_time=True, mel_template=None):
    
    if mel_template is None:
        tho_tr = bt.ThirdOctaveTransform(sr=sr, flen=flen, hlen=hlen)
        if classifier == 'PANN':
            mels_tr = bt.PANNMelsTransform(flen_tho=tho_tr.flen, device=device)
        if classifier == 'YamNet':
            mels_tr = bt.YamNetMelsTransform(flen_tho=tho_tr.flen, device=device)
        if classifier == 'default':
            mels_tr = bt.DefaultMelsTransform(sr=tho_tr.sr, flen=tho_tr.flen, hlen=tho_tr.hlen)
    else:
        tho_tr = bt.NewThirdOctaveTransform(32000, 1024, 320, 64, mel_template=mel_template, tho_freq=tho_freq, tho_time=tho_time)
        if classifier == 'PANN':
            mels_tr = bt.PANNMelsTransform(flen_tho=4096, device=device)
        if classifier == 'YamNet':
            mels_tr = bt.YamNetMelsTransform(flen_tho=4096, device=device)
        if classifier == 'default':
            mels_tr = bt.DefaultMelsTransform(sr=tho_tr.sr, flen=4096, hlen=4000)
    return(tho_tr, mels_tr)   

#count the number of parameters of a model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def sort_labels_by_score(scores, labels, top=-1):
    # create a list of tuples where each tuple contains a score and its corresponding label
    score_label_tuples = list(zip(scores, labels))
    # sort the tuples based on the score in descending order
    sorted_tuples = sorted(score_label_tuples, reverse=True)

    # extract the sorted labels from the sorted tuples
    sorted_labels = [t[1] for t in sorted_tuples]
    sorted_scores = [t[0] for t in sorted_tuples]
    
    # create a list of 1s and 0s indicating if the score is in the top 10 or not
    top_scores = sorted_scores[:top]
    top_labels = sorted_labels[:top]

    if top >= 1:
        in_top = [1 if label in top_labels else 0 for label in labels]
    else:
        in_top = None
    
    return sorted_scores, sorted_labels, in_top

def batch_logit_to_tvb(input, thresholds=[0.05, 0.06, 0.06]):
    """
    expects an input of (n_frames, labels) of numpy array
    Lorient1k normalized: 0.03, 0.15, 0.02
    Grafic normalized: 0.05, 0.06, 0.06
    """

    t = np.mean(input[:, 300] > thresholds[0])
    v = np.mean(input[:, 0] > thresholds[1])
    b = np.mean(input[:, 111] > thresholds[2])

    tvb_predictions_avg = np.array([[t,v,b]])

    return(tvb_predictions_avg)

def batch_logit_to_tvb_top(input, top_k=10):
    """
    expects an input of (n_frames, labels) of numpy array
    """

    #with numpy
    sorted_indices = np.argsort(input, axis=1)[:, ::-1]
    #with torch
    # sorted_indexes = torch.flip(torch.argsort(logits_tvb), dims=[1])

    top_indices = sorted_indices[ :, 0 : top_k]

    #307:car, 300: traffic, 0: speech, 111: bird
    #with numpy
    t = np.expand_dims((top_indices == 307).any(axis=1), axis=1)
    v = np.expand_dims((top_indices == 0).any(axis=1), axis=1)
    b = np.expand_dims((top_indices == 111).any(axis=1), axis=1)
    #with torch
    # t_label = (labels_enc_top == 300).any(dim=1).unsqueeze(dim=1)
    # v_label = (labels_enc_top == 0).any(dim=1).unsqueeze(dim=1)
    # b_label = (labels_enc_top == 111).any(dim=1).unsqueeze(dim=1)

    #with numpy
    tvb_predictions = np.concatenate((t, v, b), axis=1)
    #with torch
    # contains_values = torch.cat((t_label, v_label, b_label), dim=1).float()

    #with numpy
    tvb_predictions_avg = tvb_predictions.mean(axis=0)
    #with torch
    # labels_str_top = contains_values.mean(dim=0)

    #with numpy
    tvb_predictions_avg = np.expand_dims(tvb_predictions_avg, axis=0)        
    #with torch
    # labels_str_top = labels_str_top.unsqueeze(dim=0)
    # labels_str_top = labels_str_top.cpu().numpy()

    return(tvb_predictions_avg)

def plot_multi_spectro(x_m, fs, title='title', vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', save=False, extlmax=None):
    if isinstance(x_m[0], torch.Tensor):
        if vmin is None:
            vmin = torch.min(x_m[0])
        if vmax is None:
            vmax = torch.max(x_m[0])
    elif isinstance(x_m[0], np.ndarray):
        if vmin is None:
            vmin = np.min(x_m[0])
        if vmax is None:
            vmax = np.max(x_m[0])
    elif isinstance(x_m[0], list):
        if vmin is None:
            vmin = min(x_m[0])
        if vmax is None:
            vmax = max(x_m[0])

    exthmin = 1
    exthmax = len(x_m)
    extlmin = 0
    if extlmax==None:
        extlmax = len(x_m[0])

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20
    #mpl.use("pgf")
    # mpl.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'Times New Roman',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })

    #fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    fig, axs = plt.subplots(ncols=len(x_m), figsize=(len(x_m)*8, 5))
    #fig.subplots_adjust(wspace=1)

    for i, ax in enumerate(axs):
        if type(ylabel) is list:
            exthmin = 1
            exthmax = len(x_m[i])
            ylabel_ = ylabel[i] 
        else:
            if i == 0:
                ylabel_ = ylabel
            else:
                ylabel_ = ''
        if diff:
            im = ax.imshow(x_m[i], extent=[extlmin,extlmax,exthmin,exthmax], cmap='seismic',
                    vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        else:
            im = ax.imshow(x_m[i], extent=[extlmin,extlmax,exthmin,exthmax], cmap='inferno',
                    vmin=vmin, vmax=vmax, origin='lower', aspect='auto')

        ax.set_title(title[i])
        #ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel_)

    fig.text(0.5, 0.1, 'Time (s)', ha='center', va='center')
    
    #cbar_ax = fig.add_axes([0.06, 0.15, 0.01, 0.7])
    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Power (dB)')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('left')

    # if type(ylabel) is list:
    #     for ax, lab in zip(axs, ylabel):
    #         ax.set_ylabel(lab)
    # else:
    #     axs[0].set_ylabel(ylabel)

    #fig.tight_layout()
    #fig.tight_layout(rect=[0.1, 0.05, 1, 1], pad=2)
    fig.tight_layout(rect=[0, 0.05, 0.92, 1], pad=2)
    #fig.savefig('fig_spectro' + name + '.pdf', bbox_inches='tight', dpi=fig.dpi)
    if save:
        plt.savefig('fig_spectro' + name + '.pdf', dpi=fig.dpi, bbox_inches='tight')
    else:
        plt.show()

def roll_with_fade(audio, roll_amount, fade_in_duration=None, fade_out_duration=None):
    """
    Rolls an audio signal with a crossfade at the roll position to avoid clicks.
    
    Parameters:
        audio (numpy.ndarray): The audio signal (1D numpy array for mono, 2D for stereo).
        roll_amount (int): Number of samples to roll the audio by.
        fade_duration (int): Duration of the crossfade in samples.
        
    Returns:
        numpy.ndarray: The rolled audio signal with crossfade applied.
    """
    if fade_in_duration is None:
        fade_in_duration = len(audio) // 10

    if fade_out_duration is None:
        fade_out_duration = fade_in_duration

    # Roll the audio
    rolled_audio = np.roll(audio, roll_amount)
    
    # Create crossfade window
    fade_in = np.linspace(0, 1, fade_in_duration)
    fade_out = np.linspace(1, 0, fade_out_duration)

    # Apply crossfade at the roll boundary
    if roll_amount > fade_out_duration:
        rolled_audio[roll_amount-fade_out_duration:roll_amount] = rolled_audio[roll_amount-fade_out_duration:roll_amount] * fade_out
    if roll_amount < (len(audio) - fade_in_duration):
        rolled_audio[roll_amount:roll_amount+fade_in_duration] = rolled_audio[roll_amount:roll_amount+fade_in_duration] * fade_in
    return rolled_audio

def roll_with_crossfade(audio, roll_amount, fade_duration=None):
    """
    Rolls an audio signal with a crossfade at the roll position to avoid clicks.
    
    Parameters:
        audio (numpy.ndarray): The audio signal (1D numpy array for mono, 2D for stereo).
        roll_amount (int): Number of samples to roll the audio by.
        fade_duration (int): Duration of the crossfade in samples.
        
    Returns:
        numpy.ndarray: The rolled audio signal with crossfade applied.
    """
    if fade_duration is None:
        fade_duration = len(audio) // 10
    if roll_amount < fade_duration:
        fade_duration = roll_amount
    if roll_amount > len(audio) - fade_duration:
        fade_duration = len(audio) - roll_amount

    # Roll the audio
    rolled_audio = np.roll(audio, roll_amount)
    
    # Create crossfade window
    fade_in = np.linspace(0, 1, fade_duration)
    fade_out = np.linspace(1, 0, fade_duration)

    # Apply crossfade at the roll boundary        
    rolled_audio[roll_amount-fade_duration:roll_amount] = \
        rolled_audio[roll_amount-fade_duration:roll_amount] * fade_out + \
        rolled_audio[roll_amount:roll_amount+fade_duration][::-1] * fade_in

    return rolled_audio

def blur_audio_randomwindowsandhop_convtasnet(y, sr, winmin, winmax, hopmin_perc=0.50, hopmax_perc=0.90, threshold = 0.001, mixwin=True):

    # bundle = HDEMUCS_HIGH_MUSDB_PLUS
    bundle = CONVTASNET_BASE_LIBRI2MIX

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)
    waveform = waveform.unsqueeze(0)

    # Resample the audio if necessary
    if sr != CONVTASNET_BASE_LIBRI2MIX.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=CONVTASNET_BASE_LIBRI2MIX.sample_rate)
        waveform = resampler(waveform)

    # Add a batch and channel dimension to the waveform (expected input shape: [batch, channels, time])
    waveform = waveform.unsqueeze(0)

    # Perform source separation
    with torch.no_grad():
        separated_sources = model(waveform)

    # Join the sources to ensure the same mean and std
    joined_sources = separated_sources[0, 0, :] + separated_sources[0, 1, :]

    # Normalize the separated sources to match the original waveform's mean and std
    mean_waveform = torch.mean(waveform)
    std_waveform = torch.std(waveform)

    # Adjust the separated sources to match the waveform's mean and std
    separated_sources = (separated_sources - torch.mean(joined_sources)) / torch.std(joined_sources) * std_waveform + mean_waveform

    chunker = FullRandomHopAudioChunks(winmin=winmin, winmax=winmax, hopmin_perc=hopmin_perc, hopmax_perc=hopmax_perc)

    # The model outputs a tensor of shape [batch, num_sources, time]
    y_voice = separated_sources[0, 0, :].detach().cpu().numpy()  # First source (e.g., voice)
    y_other = separated_sources[0, 1, :].detach().cpu().numpy()  # Second source (e.g., background)

    # SHORT TERM TIME RANDOM TIME REVERSAL: chunk array into chunks of random windows, and invert the windows
    y_n_voice = chunker.chunks_with_hop(y_voice)
    # MT: to reactivate
    y_n_r_voice = [y[::-1] for y in y_n_voice]
    # y_n_r_voice = y_n_voice.copy()
    y_r_voice = chunker.concat_with_hop(y_n_r_voice)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    chunker_thresh = AudioChunks(n=round(0.100*sr), hop=round(0.080*sr))

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)

    if mixwin:
        #random shuffle of windows on y_n_r_voice
        y_n_r_voice_mean = np.sqrt(np.mean(y_n_r_voice**2))
        y_n_r_voice_b = (np.sqrt(np.mean(y_n_r_voice**2, axis=1)) > y_n_r_voice_mean*0.25).astype(int)

        # Identify segments of contiguous voice and shuffle them
        for key, group in groupby(enumerate(y_n_r_voice_b), lambda x: x[1]):
            if key == 1:
                indices = [index for index, _ in group]
                subarray = y_n_r_voice[indices]
                np.random.shuffle(subarray)
                y_n_r_voice[indices] = subarray


    y_n_t = []
    for y_i, y_i_r_voice, y_i_other in zip(y_n, y_n_r_voice, y_n_other):
        if np.sqrt(np.mean(y_i_r_voice**2)) >= threshold:
            y_n_t.append(y_i_r_voice + y_i_other)
            # y_n_t.append(y_i_r_voice)
            # y_n_t.append(y_i_other)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    return(y_combined)

def blur_audio_randomwindowsandhop_avgtime(y, sr, winmin, winmax, hopmin_perc=0.50, hopmax_perc=0.90, threshold = 0.001, mixwin=True):

    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    # bundle = CONVTASNET_BASE_LIBRI2MIX

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32).repeat(2, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    # parameters
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    # print('AAAAAAAAA')
    # print(HDEMUCS_HIGH_MUSDB_PLUS.sample_rate)

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate=sr,
    )[0]

    print(sources.shape)

    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    chunker = FullRandomHopAudioChunks(winmin=winmin, winmax=winmax, hopmin_perc=hopmin_perc, hopmax_perc=hopmax_perc)

    voice = audios['vocals']
    other = sum(audios[key] for key in audios.keys() if key != 'vocals')

    y_voice = voice.detach().cpu().numpy()[0]
    y_other = other.detach().cpu().numpy()[0]

    # SHORT TERM TIME RANDOM TIME REVERSAL: chunk array into chunks of random windows, and invert the windows
    y_n_voice = chunker.chunks_with_hop(y_voice)
    # MT: to reactivate
    y_n_r_voice = [y[::-1] for y in y_n_voice]
    # y_n_r_voice = y_n_voice.copy()
    y_r_voice = chunker.concat_with_hop(y_n_r_voice)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    chunker_thresh = AudioChunks(n=round(0.100*sr), hop=round(0.080*sr))

    #Mean over time windows
    win = 1024
    hop = int(win/4)  # No overlap
    D = np.abs(librosa.stft(y_r_voice, n_fft=win, hop_length=hop))
    time_frames = D.shape[1]

    # Calculate the window length in frames (125 ms)
    window_length_ms = 75
    window_length_frames = int((window_length_ms / 1000) * sr / hop)

    # Step 2: Average amplitudes over the 50 ms windows
    avg_amplitude = []
    for start in range(0, time_frames, window_length_frames):
        end = min(start + window_length_frames, time_frames)
        avg_amplitude.append(D[:, start:end].mean(axis=1))

    # Convert the list to a numpy array
    avg_amplitude = np.array(avg_amplitude).T

    # Step 3: Interpolate to match the original time frame dimensions
    original_times = np.arange(avg_amplitude.shape[1]) * window_length_frames
    new_times = np.arange(time_frames)
    interp_func = scipy.interpolate.interp1d(
        original_times, avg_amplitude, kind='linear', axis=1, fill_value="extrapolate"
    )
    D_interpolated = interp_func(new_times)

    # Calculate the frame index for the 2-second mark
    seconds_to_plot = 2
    frames_to_plot = int(seconds_to_plot * sr / hop)

    # Plotting the original spectrogram (D) and the interpolated spectrogram (D_interpolated) side-by-side
    plt.figure(figsize=(12, 6))

    # Original spectrogram
    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(D[:, :frames_to_plot], ref=np.max), sr=sr, hop_length=hop, y_axis='log', x_axis='time')
    plt.title("Original Spectrogram (D)")
    plt.colorbar(format="%+2.0f dB")
    plt.xlim([0, seconds_to_plot])

    # Interpolated spectrogram
    plt.subplot(1, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(D_interpolated[:, :frames_to_plot], ref=np.max), sr=sr, hop_length=hop, y_axis='log', x_axis='time')
    plt.title("Interpolated Spectrogram (D_interpolated)")
    plt.colorbar(format="%+2.0f dB")
    plt.xlim([0, seconds_to_plot])

    # Display the plots
    plt.tight_layout()
    plt.show()

    # print(D_interpolated)

    # Step 4: Reconstruct the phase using Griffin-Lim algorithm
    y_r_voice_avg = librosa.griffinlim(D_interpolated, n_iter=32, n_fft=win, hop_length=hop)

    #Mean over time windows
    # Step 1: Convert the audio file to a spectrogram
    # win = 2048
    # hop = 32
    # D = np.abs(librosa.stft(y_r_voice, n_fft=win, hop_length=hop))  # magnitude spectrogram
    # time_frames = D.shape[1]

    # # Calculate the window length in frames (125 ms)
    # window_length_ms = 32
    # window_length_frames = int((window_length_ms / 1000) * sr / hop)

    # # Step 2: Average amplitudes over 125 ms windows
    # avg_amplitude = []
    # for start in range(0, time_frames, window_length_frames):
    #     end = min(start + window_length_frames, time_frames)
    #     avg_amplitude.append(D[:, start:end].mean(axis=1))

    # # Convert list to a numpy array
    # avg_amplitude = np.array(avg_amplitude).T

    # # Step 3: Interpolate to match original time frame dimensions
    # original_times = np.arange(avg_amplitude.shape[1]) * window_length_frames
    # new_times = np.arange(time_frames)
    # interp_func = scipy.interpolate.interp1d(original_times, avg_amplitude, kind='linear', axis=1, fill_value="extrapolate")
    # D_interpolated = interp_func(new_times)

    # # Step 4: Convert the spectrogram back to audio
    # # Reconstruct the phase using the original phase
    # angle = np.angle(librosa.stft(y, n_fft=win, hop_length=hop))
    # D_reconstructed = D_interpolated * np.exp(1j * angle)

    # # Inverse Short-Time Fourier Transform
    # y_r_voice_avg = librosa.istft(D_reconstructed, hop_length=hop)

    # y_r_voice_avg = y_r_voice

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice_avg)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)

    if mixwin:
        #random shuffle of windows on y_n_r_voice
        y_n_r_voice_mean = np.sqrt(np.mean(y_n_r_voice**2))
        y_n_r_voice_b = (np.sqrt(np.mean(y_n_r_voice**2, axis=1)) > y_n_r_voice_mean*0.25).astype(int)

        # Identify segments of contiguous voice and shuffle them
        for key, group in groupby(enumerate(y_n_r_voice_b), lambda x: x[1]):
            if key == 1:
                indices = [index for index, _ in group]
                subarray = y_n_r_voice[indices]
                np.random.shuffle(subarray)
                y_n_r_voice[indices] = subarray


    y_n_t = []
    for y_i, y_i_r_voice, y_i_other in zip(y_n, y_n_r_voice, y_n_other):
        if np.sqrt(np.mean(y_i_r_voice**2)) >= threshold:
            y_n_t.append(y_i_r_voice + y_i_other)
            # y_n_t.append(y_i_r_voice)
            # y_n_t.append(y_i_other)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    return(y_combined)

def generate_gibberish_tts(length, sr):
    """
    Generate a 'blablabla' audio signal using gTTS, save it, and re-open it as a numpy array.
    """

    # Create the output directory if it doesn't exist
    output_dir = "./temp"
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, "temp_audio.wav")

    # Generate TTS audio and save to a file
    tts = gTTS("stew purple instant ugliest interfere silicon tenor thoughtful resort evocation wagon lapdog rooster sigh freighter path comma zonked slapstick" * int(length), lang="en")
    tts.save(temp_file)

    # Load the audio file and resample to the desired sample rate
    waveform, original_sr = torchaudio.load(temp_file)

    # Resample if necessary
    if original_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=sr)
        waveform = resampler(waveform)

    # Convert waveform to numpy array and normalize to [-1, 1]
    waveform = waveform.numpy().flatten()
    waveform /= np.max(np.abs(waveform))

    # Tile the signal to match the required length
    target_samples = int(length * sr)
    tiled_signal = np.tile(waveform, (target_samples // len(waveform) + 1))[:target_samples]

    return tiled_signal

def calculate_energy_profile(signal, window_size, hop_size):
    """
    Calculate the energy profile of a signal using overlapping windows.
    """
    n_windows = (len(signal) - window_size) // hop_size + 1
    energy_profile = np.zeros(len(signal))

    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size
        window_energy = np.sqrt(np.mean(signal[start:end] ** 2))
        energy_profile[start:end] = window_energy

    # Handle any remaining samples at the end
    if end < len(signal):
        window_energy = np.sqrt(np.mean(signal[end:] ** 2))
        energy_profile[end:] = window_energy

    return energy_profile

def apply_energy_profile(tts_signal, orig_energy_profile, tts_energy_profile, window_size, hop_size):
    """
    Apply the energy profile of the original signal to the TTS signal.
    """

    modulated_signal = np.zeros_like(tts_signal)
    n_windows = (len(tts_signal) - window_size) // hop_size + 1

    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size

        # Avoid division by zero for TTS energy
        tts_energy = tts_energy_profile[start:end] if np.sum(tts_energy_profile[start:end]) > 0 else 1.0

        # Energy ratio for the current window
        energy_ratio = orig_energy_profile[start:end] / tts_energy if np.mean(orig_energy_profile[start:end]) > 0.01 else 0

        # Modulate the TTS signal
        modulated_signal[start:end] = tts_signal[start:end] * energy_ratio

    # Handle any remaining samples at the end
    if end < len(tts_signal):
        tts_energy = tts_energy_profile[end:] if np.sum(tts_energy_profile[end:]) > 0 else 1.0
        energy_ratio = orig_energy_profile[end:] / tts_energy
        modulated_signal[end:] = tts_signal[end:] * energy_ratio

    return modulated_signal

def blur_audio_tts(y, sr):
    """
    reverse: "random", "forward", "backward"
    """
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    # bundle = CONVTASNET_BASE_LIBRI2MIX

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32).repeat(2, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    # parameters
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate=sr,
    )[0]

    print(sources.shape)

    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))

    voice = audios['vocals']
    other = sum(audios[key] for key in audios.keys() if key != 'vocals')

    y_voice = voice.detach().cpu().numpy()[0]
    y_other = other.detach().cpu().numpy()[0]
    # Generate the TTS signal
    tts_signal = generate_gibberish_tts(length=y_voice.shape[0] / sr, sr=sr)

    # Calculate energy profile of the original voice
    window_size = int(0.025 * sr)  # 25 ms window
    hop_size = int(0.01 * sr)     # 10 ms hop
    orig_energy_profile = calculate_energy_profile(y_voice, window_size, hop_size)
    tts_energy_profile = calculate_energy_profile(tts_signal, window_size, hop_size)

    # Apply the energy profile to the TTS signal
    modulated_tts = apply_energy_profile(tts_signal, orig_energy_profile, tts_energy_profile, window_size, hop_size)

    # Combine the modulated TTS signal with the other sources
    y_combined = y_other + modulated_tts

    return y_combined


def blur_audio_randomwindowsandhop_phaseshift_shuffle_with_classif(y, sr, winmin, winmax, hopmin_perc=0.50, hopmax_perc=0.90, threshold = 0.001, mixwin=True, randphase=True, reverse="forward",
                                                      device="cpu", classif_model=None, classif_sr=None, classif_speech_index=None, classif_speech_ths=0.5):
    """
    reverse: "random", "forward", "backward"
    """
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    # bundle = CONVTASNET_BASE_LIBRI2MIX

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32).repeat(2, 1)
    device = device
    waveform = waveform.to(device)
    model = model.to(device)

    # parameters
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate=sr,
    )[0]

    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    chunker = FullRandomHopAudioChunksShuffle(winmin=winmin, winmax=winmax, hopmin_perc=hopmin_perc, hopmax_perc=hopmax_perc)

    voice = audios['vocals']
    other = sum(audios[key] for key in audios.keys() if key != 'vocals')

    y_voice = voice.detach().cpu().numpy()[0]
    y_other = other.detach().cpu().numpy()[0]

    # SHORT TERM TIME RANDOM TIME REVERSAL: chunk array into chunks of random windows, shuffle them and invert the windows
    y_n_voice = chunker.chunks_with_hop(y_voice, mixwin=mixwin)

    if reverse == "backward":
        order = [-1 for _ in y_n_voice]
    elif reverse == "forward":
        order = [1 for _ in y_n_voice]
    else:
        order = [random.choice([-1, 1]) for _ in y_n_voice]

    if randphase:
        y_n_r_voice = [roll_with_crossfade(y[::order_i], random.randint(len(y) // 10, len(y) - len(y) // 10), 
                                    fade_duration=random.randint(len(y) // 10, len(y) // 5)) \
                    for _, (y, order_i) in enumerate(zip(y_n_voice, order))]
    else:
        y_n_r_voice = [y[::order_i] for _, (y, order_i) in enumerate(zip(y_n_voice, order))]

    y_r_voice = chunker.concat_with_hop(y_n_r_voice)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    # chunker_thresh = AudioChunks(n=round(0.500*sr), hop=round(0.440*sr))
    chunker_thresh = AudioChunks(n=round(0.5*sr), hop=round(0.440*sr))

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)
    y_n_is_speech = np.zeros(len(y_n), dtype=bool)
        
    # Resample the audio to 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=classif_sr)
    y_beats = resampler(torch.tensor(y_n, dtype=torch.float32))

    # Normalize y_beats to max absolute value
    y_beats = y_beats / torch.max(torch.abs(y_beats))

    # Pad y_beats with zeros until it reaches 0.5s
    target_length = int(0.5 * 16000)  # 0.5 seconds at 16kHz
    if y_beats.shape[0] < target_length:
        padding = target_length - y_beats.shape[0]
        y_beats = torch.nn.functional.pad(y_beats, (0, padding))

    # Predict with the model
    with torch.no_grad():
        input_tensor = torch.tensor(y_beats).to(device)
        padding_mask = torch.zeros(input_tensor.size(0), input_tensor.size(1)).bool().to(device)  # Assuming no padding for the single audio
        speech_logit = classif_model.extract_features(input_tensor, padding_mask=padding_mask)[0][:, classif_speech_index]

    y_n_is_speech = speech_logit > classif_speech_ths

    # for i, y_i_n in enumerate(y_n):
        
    #     # Resample the audio to 16kHz
    #     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=classif_sr)
    #     y_beats = resampler(torch.tensor(y_i_n, dtype=torch.float32))

    #     # Normalize y_beats to max absolute value
    #     y_beats = y_beats / torch.max(torch.abs(y_beats))

    #     # Pad y_beats with zeros until it reaches 0.5s
    #     target_length = int(0.5 * 16000)  # 0.5 seconds at 16kHz
    #     if y_beats.shape[0] < target_length:
    #         padding = target_length - y_beats.shape[0]
    #         y_beats = torch.nn.functional.pad(y_beats, (0, padding))

    #     # Convert audio to tensor
    #     input_tensor = torch.tensor(y_beats).unsqueeze(0).to(device)  # Add batch dimension
    #     padding_mask = torch.zeros(1, input_tensor.size(1)).bool().to(device)

    #     # Predict with the model
    #     with torch.no_grad():
    #         speech_logit = classif_model.extract_features(input_tensor, padding_mask=padding_mask)[0][0][classif_speech_index]
    #     y_n_is_speech[i] = speech_logit > classif_speech_ths

    y_n_t = []

    for y_i, y_i_r_voice, y_i_other, y_i_is_speech in zip(y_n, y_n_r_voice, y_n_other, y_n_is_speech):
        temp = y_i_r_voice + y_i_other
        if y_i_is_speech:
            y_n_t.append(y_i_r_voice + y_i_other)
            # if np.sqrt(np.mean(y_i_r_voice**2)) >= threshold:
            #     y_n_t.append(y_i_r_voice + y_i_other)
            #     # y_n_t.append(y_i_r_voice)
            #     # y_n_t.append(y_i_other)
            # else:
            #     y_n_t.append(y_i)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    if len(y_combined) < len(y):
        y_combined = np.concatenate((y_combined, y[len(y_combined):]))



    return(y_combined)


def blur_audio_randomwindowsandhop_phaseshift_shuffle(y, sr, winmin, winmax, hopmin_perc=0.50, hopmax_perc=0.90, threshold = 0.001, mixwin=True, randphase=True, reverse="forward",
                                                      device="cpu"):
    """
    reverse: "random", "forward", "backward"
    """
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    # bundle = CONVTASNET_BASE_LIBRI2MIX

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32).repeat(2, 1)
    device = device
    waveform = waveform.to(device)
    model = model.to(device)

    # parameters
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate=sr,
    )[0]

    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    chunker = FullRandomHopAudioChunksShuffle(winmin=winmin, winmax=winmax, hopmin_perc=hopmin_perc, hopmax_perc=hopmax_perc)

    voice = audios['vocals']
    other = sum(audios[key] for key in audios.keys() if key != 'vocals')

    y_voice = voice.detach().cpu().numpy()[0]
    y_other = other.detach().cpu().numpy()[0]

    # SHORT TERM TIME RANDOM TIME REVERSAL: chunk array into chunks of random windows, shuffle them and invert the windows
    y_n_voice = chunker.chunks_with_hop(y_voice, mixwin=mixwin)

    if reverse == "backward":
        order = [-1 for _ in y_n_voice]
    elif reverse == "forward":
        order = [1 for _ in y_n_voice]
    else:
        order = [random.choice([-1, 1]) for _ in y_n_voice]

    if randphase:
        y_n_r_voice = [roll_with_crossfade(y[::order_i], random.randint(len(y) // 10, len(y) - len(y) // 10), 
                                    fade_duration=random.randint(len(y) // 10, len(y) // 5)) \
                    for _, (y, order_i) in enumerate(zip(y_n_voice, order))]
    else:
        y_n_r_voice = [y[::order_i] for _, (y, order_i) in enumerate(zip(y_n_voice, order))]

    y_r_voice = chunker.concat_with_hop(y_n_r_voice)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    chunker_thresh = AudioChunks(n=round(0.500*sr), hop=round(0.440*sr))

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)

    y_n_t = []
    for y_i, y_i_r_voice, y_i_other in zip(y_n, y_n_r_voice, y_n_other):
        if np.sqrt(np.mean(y_i_r_voice**2)) >= threshold:
            y_n_t.append(y_i_r_voice + y_i_other)
            # y_n_t.append(y_i_r_voice)
            # y_n_t.append(y_i_other)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    return(y_combined)


def blur_audio_randomwindowsandhop_phaseshift(y, sr, winmin, winmax, hopmin_perc=0.50, hopmax_perc=0.90, threshold = 0.001, mixwin=True, randphase=True):

    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    # bundle = CONVTASNET_BASE_LIBRI2MIX

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32).repeat(2, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    # parameters
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    # print('AAAAAAAAA')
    # print(HDEMUCS_HIGH_MUSDB_PLUS.sample_rate)

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate=sr,
    )[0]

    print(sources.shape)

    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    chunker = FullRandomHopAudioChunks(winmin=winmin, winmax=winmax, hopmin_perc=hopmin_perc, hopmax_perc=hopmax_perc)

    voice = audios['vocals']
    other = sum(audios[key] for key in audios.keys() if key != 'vocals')

    y_voice = voice.detach().cpu().numpy()[0]
    y_other = other.detach().cpu().numpy()[0]

    # SHORT TERM TIME RANDOM TIME REVERSAL: chunk array into chunks of random windows, and invert the windows
    y_n_voice = chunker.chunks_with_hop(y_voice)
    # MT: to reactivate
    # y_n_r_voice = [roll_with_fade(y[::-1], (len(y) // 4) + random.randint(1, len(y) // 4), 
    #                               fade_in_duration=random.randint(len(y) // 10, len(y) // 5),
    #                               fade_out_duration=random.randint(len(y) // 10, len(y) // 5)) \
    #                for y in y_n_voice]
    # print(len(y))
    # print(random.randint(1, (len(y) // 2) - (len(y) // 10)))
    # print(random.randint(len(y) // 5, len(y) // 3))

    if randphase:
        y_n_r_voice = [roll_with_crossfade(y[::-1], random.randint(len(y) // 10, len(y) - len(y) // 10), 
                                    fade_duration=random.randint(len(y) // 10, len(y) // 5)) \
                    for y in y_n_voice]
    else:
        y_n_r_voice = [y[::-1] for y in y_n_voice]
    # y_n_r_voice = y_n_voice.copy()
    y_r_voice = chunker.concat_with_hop(y_n_r_voice)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    chunker_thresh = AudioChunks(n=round(0.500*sr), hop=round(0.440*sr))

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)

    if mixwin:
        interval_samples = 20

        # Shuffling 100ms windows within each 5-second interval
        for interval_start in range(0, len(y_n_r_voice), interval_samples):
            interval_end = min(interval_start + interval_samples, len(y_n_r_voice))
            interval_indices = np.arange(interval_start, interval_end)
            
            subarray = y_n_r_voice[interval_indices]
            np.random.shuffle(subarray)
            y_n_r_voice[interval_indices] = subarray

    y_n_t = []
    for y_i, y_i_r_voice, y_i_other in zip(y_n, y_n_r_voice, y_n_other):
        if np.sqrt(np.mean(y_i_r_voice**2)) >= threshold:
            y_n_t.append(y_i_r_voice + y_i_other)
            # y_n_t.append(y_i_r_voice)
            # y_n_t.append(y_i_other)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    return(y_combined)


def blur_audio_randomwindowsandhop_fixmixwin(y, sr, winmin, winmax, hopmin_perc=0.50, hopmax_perc=0.90, threshold = 0.001, mixwin=True):

    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    # bundle = CONVTASNET_BASE_LIBRI2MIX

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32).repeat(2, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    # parameters
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    # print('AAAAAAAAA')
    # print(HDEMUCS_HIGH_MUSDB_PLUS.sample_rate)

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate=sr,
    )[0]

    print(sources.shape)

    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    chunker = FullRandomHopAudioChunks(winmin=winmin, winmax=winmax, hopmin_perc=hopmin_perc, hopmax_perc=hopmax_perc)

    voice = audios['vocals']
    other = sum(audios[key] for key in audios.keys() if key != 'vocals')

    y_voice = voice.detach().cpu().numpy()[0]
    y_other = other.detach().cpu().numpy()[0]

    # SHORT TERM TIME RANDOM TIME REVERSAL: chunk array into chunks of random windows, and invert the windows
    y_n_voice = chunker.chunks_with_hop(y_voice)
    # MT: to reactivate
    y_n_r_voice = [y[::-1] for y in y_n_voice]
    # y_n_r_voice = y_n_voice.copy()
    y_r_voice = chunker.concat_with_hop(y_n_r_voice)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    chunker_thresh = AudioChunks(n=round(0.500*sr), hop=round(0.440*sr))

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)

    if mixwin:
        interval_samples = 20

        # Shuffling 100ms windows within each 5-second interval
        for interval_start in range(0, len(y_n_r_voice), interval_samples):
            interval_end = min(interval_start + interval_samples, len(y_n_r_voice))
            interval_indices = np.arange(interval_start, interval_end)
            
            subarray = y_n_r_voice[interval_indices]
            np.random.shuffle(subarray)
            y_n_r_voice[interval_indices] = subarray

    y_n_t = []
    for y_i, y_i_r_voice, y_i_other in zip(y_n, y_n_r_voice, y_n_other):
        if np.sqrt(np.mean(y_i_r_voice**2)) >= threshold:
            y_n_t.append(y_i_r_voice + y_i_other)
            # y_n_t.append(y_i_r_voice)
            # y_n_t.append(y_i_other)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    return(y_combined)


def blur_audio_randomwindowsandhop(y, sr, winmin, winmax, hopmin_perc=0.50, hopmax_perc=0.90, threshold = 0.001, mixwin=True):

    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    # bundle = CONVTASNET_BASE_LIBRI2MIX

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32).repeat(2, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    # parameters
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    # print('AAAAAAAAA')
    # print(HDEMUCS_HIGH_MUSDB_PLUS.sample_rate)

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate=sr,
    )[0]

    print(sources.shape)

    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    chunker = FullRandomHopAudioChunks(winmin=winmin, winmax=winmax, hopmin_perc=hopmin_perc, hopmax_perc=hopmax_perc)

    voice = audios['vocals']
    other = sum(audios[key] for key in audios.keys() if key != 'vocals')

    y_voice = voice.detach().cpu().numpy()[0]
    y_other = other.detach().cpu().numpy()[0]

    # SHORT TERM TIME RANDOM TIME REVERSAL: chunk array into chunks of random windows, and invert the windows
    y_n_voice = chunker.chunks_with_hop(y_voice)
    # MT: to reactivate
    y_n_r_voice = [y[::-1] for y in y_n_voice]
    # y_n_r_voice = y_n_voice.copy()
    y_r_voice = chunker.concat_with_hop(y_n_r_voice)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    chunker_thresh = AudioChunks(n=round(0.100*sr), hop=round(0.080*sr))

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)

    if mixwin:
        #random shuffle of windows on y_n_r_voice
        y_n_r_voice_mean = np.sqrt(np.mean(y_n_r_voice**2))
        y_n_r_voice_b = (np.sqrt(np.mean(y_n_r_voice**2, axis=1)) > y_n_r_voice_mean*0.25).astype(int)

        # Identify segments of contiguous voice and shuffle them
        for key, group in groupby(enumerate(y_n_r_voice_b), lambda x: x[1]):
            if key == 1:
                indices = [index for index, _ in group]
                subarray = y_n_r_voice[indices]
                np.random.shuffle(subarray)
                y_n_r_voice[indices] = subarray


    y_n_t = []
    for y_i, y_i_r_voice, y_i_other in zip(y_n, y_n_r_voice, y_n_other):
        if np.sqrt(np.mean(y_i_r_voice**2)) >= threshold:
            y_n_t.append(y_i_r_voice + y_i_other)
            # y_n_t.append(y_i_r_voice)
            # y_n_t.append(y_i_other)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    return(y_combined)

def blur_audio_randomwindows(y, sr, winmin, winmax, hop_perc=0.90, threshold = 0.001, mixwin=True):

    bundle = HDEMUCS_HIGH_MUSDB_PLUS

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32).repeat(2, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    # parameters
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate=sr,
    )[0]
    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    chunker = RandomHopAudioChunks(winmin=winmin, winmax=winmax, hop_perc=hop_perc)

    voice = audios['vocals']
    other = sum(audios[key] for key in audios.keys() if key != 'vocals')

    y_voice = voice.detach().cpu().numpy()[0]
    y_other = other.detach().cpu().numpy()[0]

    # SHORT TERM TIME RANDOM TIME REVERSAL: chunk array into chunks of random windows, and invert the windows
    y_n_voice = chunker.chunks_with_hop(y_voice)
    y_n_r_voice = [y[::-1] for y in y_n_voice]

    y_r_voice = chunker.concat_with_hop(y_n_r_voice)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    chunker_thresh = AudioChunks(n=round(0.100*sr), hop=round(0.080*sr))

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)

    if mixwin:
        #random shuffle of windows on y_n_r_voice
        y_n_r_voice_mean = np.sqrt(np.mean(y_n_r_voice**2))
        y_n_r_voice_b = (np.sqrt(np.mean(y_n_r_voice**2, axis=1)) > y_n_r_voice_mean*0.25).astype(int)

        # Identify segments of contiguous voice and shuffle them
        for key, group in groupby(enumerate(y_n_r_voice_b), lambda x: x[1]):
            if key == 1:
                indices = [index for index, _ in group]
                subarray = y_n_r_voice[indices]
                # MT: to reactivate
                np.random.shuffle(subarray)
                y_n_r_voice[indices] = subarray

    y_n_t = []
    for y_i, y_i_r_voice, y_i_other in zip(y_n, y_n_r_voice, y_n_other):
        if np.sqrt(np.mean(y_i_r_voice**2)) > threshold:
            y_n_t.append(y_i_r_voice + y_i_other)
            # y_n_t.append(y_i_other)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    return(y_combined)

def blur_audio(y, sr, n_s=0.400, hop_s=0.398, threshold = 0.001):

    bundle = HDEMUCS_HIGH_MUSDB_PLUS

    model = bundle.get_model()
    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    waveform = torch.tensor(y, dtype=torch.float32).repeat(2, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    # parameters
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate=sr,
    )[0]
    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    chunker = AudioChunks(n=round(n_s*sr), hop=round(hop_s*sr))

    voice = audios['vocals']
    other = sum(audios[key] for key in audios.keys() if key != 'vocals')

    y_voice = voice.detach().cpu().numpy()[0]
    y_other = other.detach().cpu().numpy()[0]

    y_n = chunker.chunks_with_hop(y)
    y_n_voice = chunker.chunks_with_hop(y_voice)
    y_n_other = chunker.chunks_with_hop(y_other)

    # SHORT TERM TIME REVERSAL: 3 elements out of 4 are reversed. Keeping a fourth one not reversed gives a feeling of something more natural
    # y_n_r_voice = np.array([y if (i // 2) % 4 == 0 else y[::-1] for i, y in enumerate(y_n_voice)])
    # Version with simple time reversal
    y_n_r_voice = np.array([y[::-1] for y in y_n_voice])

    # Threshold to decide if we should use reversed vocals
    # threshold = 0.001  # Adjust this value based on your needs

    y_n_t = []
    y_n_r_t_voice = []
    y_n_t_voice = []
    y_n_r_t_other = []
    for original, original_v, original_o, reversed_v in zip(y_n, y_n_voice, y_n_other, y_n_r_voice):
        if np.sqrt(np.mean(original_v**2)) > threshold:  # Check RMS level
            y_n_r_t_voice.append(reversed_v)
            y_n_t_voice.append(original_v)
            y_n_r_t_other.append(original_o)
            y_n_t.append(np.zeros_like(original))
        else:
            y_n_r_t_voice.append(np.zeros_like(reversed_v))
            y_n_r_t_other.append(np.zeros_like(original_o))
            y_n_t_voice.append(np.zeros_like(original_v))
            y_n_t.append(original)

    y_n_t = np.array(y_n_t)
    y_n_r_t_voice = np.array(y_n_r_t_voice)
    y_n_r_t_other = np.array(y_n_r_t_other)
    y_n_t_voice = np.array(y_n_t_voice)

    y_n_combined = y_n_t + y_n_r_t_voice + y_n_r_t_other
    y_combined = chunker.concat_with_hop(y_n_combined)

    # Combine the blurred voice with the accompaniment
    # min_length = min(len(y_r_voice), len(y_other))
    # y_r_voice = y_r_voice[:min_length]
    # y_other = y_other[:min_length]
    # y_blurred = y_r_voice + y_other
    return(y_combined)

def separate_sources(
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
    sample_rate=32000,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

def separate_sources_env(
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
    sample_rate=32000,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    mixture = mix.reshape(1, 1, -1)
    estimated_sources = model(mixture)
    return estimated_sources