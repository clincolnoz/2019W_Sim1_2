# -*- coding: utf-8 -*-
import click
import logging
import tkinter as tk
from PIL import ImageTk, Image
from pathlib import Path


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn intermediate data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    # logger.info('Making final data set from raw data')

    root = tk.Tk()
    # Code to add widgets will go here...
    # frame
    frame = tk.Frame(root)
    frame.pack()
    # buttons
    # f = kermit, t = talking, g = no kermit
    # TODO(Discuss about Kermit talking with Craig)
    kermit_btn = tk.Button(frame, text='kermit')
    kermit_btn.pack(side="right")
    no_kermit_btn = tk.Button(frame, text='no kermit')
    no_kermit_btn.pack(side="right")
    # image
    canvas = tk.Canvas(root, width=700, height=450)
    canvas.pack()
    # TODO(Implent to iterate over all images)
    img = ImageTk.PhotoImage(Image.open(str(input_filepath) + "frame1.jpg"))
    canvas.create_image(20, 20, anchor="nw", image=img)
    # start
    root.mainloop()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()

