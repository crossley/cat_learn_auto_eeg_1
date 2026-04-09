# -*- coding: utf-8 -*-
"""
Single-phase categorization task (PsychoPy-only) with veridical feedback and EEG triggers.
Maintains the original state-machine style and adds 'day' (repeated-measures).
"""

import os, sys
import numpy as np
import pandas as pd

from util_func import *

from psychopy import visual, core, event, logging
from psychopy.hardware import keyboard

# --------------------------- EEG (Parallel Port) helper ---------------------------
# Flip-locked rising edges; non-blocking clear to zero a few ms later.
EEG_ENABLED = True
EEG_PORT_ADDRESS = '0x3FD8'   # <-- change to your lab's address (e.g., '0x0378' or int)
EEG_DEFAULT_PULSE_MS = 10

TRIG = {
    "EXP_START":   10,
    "ITI_ONSET":   11,
    "STIM_ONSET_A":20,
    "STIM_ONSET_B":21,
    "RESP_A":      30,
    "RESP_B":      31,
    "FB_COR":      40,
    "FB_INC":      41,
    "EXP_END":     15,
}

class EEGPort:
    def __init__(self, win, address=EEG_PORT_ADDRESS, enabled=EEG_ENABLED, default_ms=EEG_DEFAULT_PULSE_MS):
        self.win = win
        self.enabled = enabled
        self.default_ms = default_ms
        self._port = None
        self._clear_at = None  # (global time in seconds) when to clear to zero
        if not self.enabled:
            return
        try:
            from psychopy import parallel
            self._port = parallel.ParallelPort(address=address)
        except Exception as e:
            print(f"[EEG] Parallel port unavailable ({e}). Running without triggers.")
            self.enabled = False
            self._port = None

    def flip_pulse(self, code, width_ms=None, global_clock=None):
        """Schedule a flip-locked pulse: set code on next win.flip, clear after width_ms."""
        if not (self.enabled and self._port):
            return
        width_ms = self.default_ms if width_ms is None else width_ms
        # rising edge exactly on next flip:
        self.win.callOnFlip(self._port.setData, int(code) & 0xFF)
        # schedule a timed clear to 0 after the flip:
        if global_clock is not None:
            # record when to clear (relative to global clock)
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def pulse_now(self, code, width_ms=None, global_clock=None):
        """Immediate pulse (not flip-locked) -- useful for response events."""
        if not (self.enabled and self._port):
            return
        width_ms = self.default_ms if width_ms is None else width_ms
        self._port.setData(int(code) & 0xFF)
        if global_clock is not None:
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def update(self, global_clock=None):
        """Call every frame: clears the port to 0 if a pulse has expired."""
        if not (self.enabled and self._port):
            return
        if self._clear_at is not None and global_clock is not None:
            if global_clock.getTime() >= self._clear_at:
                self._port.setData(0)
                self._clear_at = None

    def close(self):
        try:
            if self._port:
                self._port.setData(0)
        except Exception:
            pass

# ----------------------------------------------------------------------------------

if __name__ == "__main__":

    # --------------------------- Subject / Day handling ---------------------------
    subject = 1   # set via CLI/dialog as needed
    day = 1       # repeated-measures day/session (1,2,3,...)

    dir_data = "../data"
    os.makedirs(dir_data, exist_ok=True)
    f_name = f"sub_{subject:03d}_day_{day:02d}_data.csv"
    full_path = os.path.join(dir_data, f_name)

    if os.path.exists(full_path):
        print(f"File {f_name} already exists. Aborting.")
        sys.exit()

    # --------------------------- Stimulus set (single phase) ----------------------
    ds = make_stim_cats()                  # expects columns: x,y,xt,yt,cat
    ds = ds.sample(frac=1).reset_index(drop=True)
    n_trial = ds.shape[0]

    # --------------------------- Display / geometry -------------------------------
    # Keep your pixel-based style. We'll set units='pix' and convert sf to cycles/pixel.
    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54
    size_cm = 5
    size_px = int(size_cm * px_per_cm)
    
    win = visual.Window(
    size=(1920, 1080),  # <-- pick something reasonable for your display
    fullscr=True,
    units='pix',
    color=(0.494, 0.494, 0.494),
    colorSpace='rgb',
    winType='pyglet',
    useRetina=True,      # on Mac/Retina; set False if you prefer logical pixels
    waitBlanking=True    # vsync; good for timing
    )
    
    win.mouseVisible = False
    frame_rate = win.getActualFrameRate()
    print(f"[Info] Frame rate: {frame_rate}")

    # Coordinates for center (not strictly needed with PsychoPy)
    center_x, center_y = 0, 0

    # --------------------------- Stim objects ------------------------------------
    # Fixation cross (two lines)
    fix_h = visual.Line(win, start=(0, -10), end=(0, 10), lineColor='white', lineWidth=4)
    fix_v = visual.Line(win, start=(-10, 0), end=(10, 0), lineColor='white', lineWidth=4)

    # Instruction text for INIT
    init_text = visual.TextStim(win, text="Please press the space bar to begin",
                                color='white', height=32)

    finished_text = visual.TextStim(win, text="You finished! Thank you for participating!",
                                    color='white', height=32)

    # Grating stimulus (parameters set per-trial)
    grating = visual.GratingStim(
        win,
        tex='sin', mask=None,
        size=(size_px, size_px),
        units='pix',
        sf=0.02,    # placeholder; set per trial as cycles/pixel
        ori=0.0
    )

    # Feedback ring (outline circle)
    fb_ring = visual.Circle(
        win,
        radius=(size_px // 2 + 10),
        edges=128,
        fillColor=None,
        lineColor='white',
        lineWidth=5,
        units='pix',
        pos=(center_x, center_y)
    )

    # Keyboard
    kb = keyboard.Keyboard()
    default_kb = keyboard.Keyboard()

    # Clocks
    global_clock = core.Clock()  # monotonic time from experiment start
    state_clock = core.Clock()
    stim_clock = core.Clock()

    # --------------------------- EEG init ----------------------------------------
    eeg = EEGPort(win)

    # --------------------------- State machine setup ------------------------------
    time_state = 0.0
    state_current = "state_init"
    state_entry = True

    resp = -1
    rt = -1
    trial = -1

    # Record keeping (added 'day')
    trial_data = {
        'subject': [],
        'day': [],
        'trial': [],
        'cat': [],
        'x': [],
        'y': [],
        'xt': [],
        'yt': [],
        'resp': [],
        'rt': [],
        'fb': []
    }

    # --------------------------- Main loop ---------------------------------------
    running = True
    while running:

        # Escape to quit anytime
        if default_kb.getKeys(keyList=['escape'], waitRelease=False):
            running = False
            break

        # Per-frame EEG housekeeping (clear any expired pulses)
        eeg.update(global_clock)

        # --------------------- STATE: INIT ---------------------
        if state_current == "state_init":
            if state_entry:
                state_clock.reset()
                win.color = (0.494, 0.494, 0.494)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0  # ms
            init_text.draw()

            # Start on SPACE
            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True)
            if keys:
                eeg.flip_pulse(TRIG["EXP_START"], global_clock=global_clock)
                state_current = "state_iti"
                state_entry = True

            win.flip()

        # --------------------- STATE: FINISHED ---------------------
        elif state_current == "state_finished":
            if state_entry:
                eeg.flip_pulse(TRIG["EXP_END"], global_clock=global_clock)
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            finished_text.draw()
            win.flip()

        # --------------------- STATE: ITI ---------------------
        elif state_current == "state_iti":
            if state_entry:
                state_clock.reset()
                eeg.flip_pulse(TRIG["ITI_ONSET"], global_clock=global_clock)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            # Draw fixation
            fix_h.draw()
            fix_v.draw()

            # Fixed 1000 ms ITI
            if time_state > 1000:
                resp = -1
                rt = -1
                state_clock.reset()
                trial += 1
                if trial >= n_trial:
                    state_current = "state_finished"
                    state_entry = True
                else:
                    # Prepare this trial's params
                    sf_cycles_per_cm = float(ds['xt'].iloc[trial])   # cycles/cm from your space
                    sf_cycles_per_pix = sf_cycles_per_cm / px_per_cm  # convert to cycles/pixel
                    ori_deg = float(ds['yt'].iloc[trial])
                    cat = ds['cat'].iloc[trial]  # "A" or "B"

                    grating.sf = sf_cycles_per_pix
                    grating.ori = ori_deg
                    grating.pos = (center_x, center_y)

                    # prepare for response collection aligned to stim onset
                    kb.clearEvents()
                    state_current = "state_stim"
                    state_entry = True

            win.flip()

        # --------------------- STATE: STIM ---------------------
        elif state_current == "state_stim":
            if state_entry:
                # Flip-locked stim-onset code (category-specific)
                trig = TRIG["STIM_ONSET_A"] if cat == "A" else TRIG["STIM_ONSET_B"]
                eeg.flip_pulse(trig, global_clock=global_clock)

                state_clock.reset()
                stim_clock.reset()
                # Align keyboard RTs to stimulus flip:
                win.callOnFlip(kb.clock.reset)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            # Draw grating
            grating.draw()

            # Collect response: D -> A, K -> B
            keys = kb.getKeys(keyList=['d', 'k'], waitRelease=False)
            if keys:
                k = keys[-1]  # last key
                rt = k.rt * 1000.0  # seconds -> ms
                if k.name == 'd':
                    resp_label = "A"
                    eeg.pulse_now(TRIG["RESP_A"], global_clock=global_clock)
                else:
                    resp_label = "B"
                    eeg.pulse_now(TRIG["RESP_B"], global_clock=global_clock)

                fb = "Correct" if (cat == resp_label) else "Incorrect"
                resp = resp_label

                state_clock.reset()
                state_current = "state_feedback"
                state_entry = True

            win.flip()

        # --------------------- STATE: FEEDBACK ---------------------
        elif state_current == "state_feedback":
            if state_entry:
                # Draw only the feedback ring (not the grating), like your pygame version.
                fb_ring.lineColor = 'green' if fb == "Correct" else 'red'
                eeg.flip_pulse(TRIG["FB_COR"] if fb == "Correct" else TRIG["FB_INC"],
                               global_clock=global_clock)
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            fb_ring.draw()

            # 1000 ms feedback, then log and move on
            if time_state > 1000:
                trial_data['subject'].append(subject)
                trial_data['day'].append(day)
                trial_data['trial'].append(trial)
                trial_data['cat'].append(ds['cat'].iloc[trial])
                trial_data['x'].append(ds['x'].iloc[trial])
                trial_data['y'].append(ds['y'].iloc[trial])
                trial_data['xt'].append(ds['xt'].iloc[trial])
                trial_data['yt'].append(ds['yt'].iloc[trial])
                trial_data['resp'].append(resp)
                trial_data['rt'].append(rt)
                trial_data['fb'].append(fb)

                # Incremental save (same behavior as your code)
                pd.DataFrame(trial_data).to_csv(full_path, index=False)

                state_current = "state_iti"
                state_entry = True
                resp = -1
                rt = -1

            win.flip()

    # --------------------------- Cleanup ------------------------------------------
    eeg.close()
    win.close()
    core.quit()
