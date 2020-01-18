# -*- coding: utf-8 -*-
import click
import logging
import tkinter as tk
import shutil
from PIL import ImageTk, Image
from pathlib import Path
from os import listdir
from tkinter import messagebox


# GLOBALS
# image counter
IMG_i = 0


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn intermediate data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    # tkinter base
    root = tk.Tk()

    # frame
    frame = tk.Frame(root)
    frame.pack()

    # get all frames in interim
    frames = listdir(str(input_filepath))

    # load first image
    img = ImageTk.PhotoImage(Image.open(input_filepath + frames[0]))
    panel = tk.Label(root, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")

    def show_next_img(i: int):
        if i == len(frames):
            messagebox.showinfo(
                "Information", "You're done labeling your input folder!"
            )
            root.destroy()
        new_img = ImageTk.PhotoImage(Image.open(input_filepath + frames[i]))
        panel.configure(image=new_img)
        panel.image = new_img
        global IMG_i
        IMG_i += 1

    # buttons and callbacks
    def kermit_callback(i: int):
        shutil.copy(input_filepath + frames[i], output_filepath + "kermit/" + frames[i])
        i += 1
        show_next_img(i)

    def no_kermit_callback(i: int):
        shutil.copy(
            input_filepath + frames[i], output_filepath + "no_kermit/" + frames[i]
        )
        i += 1
        show_next_img(i)

    kermit_btn = tk.Button(frame, text="kermit", command=lambda: kermit_callback(IMG_i))
    kermit_btn.pack(side="right")
    no_kermit_btn = tk.Button(
        frame, text="no kermit", command=lambda: no_kermit_callback(IMG_i)
    )
    no_kermit_btn.pack(side="right")

    # start
    root.mainloop()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
