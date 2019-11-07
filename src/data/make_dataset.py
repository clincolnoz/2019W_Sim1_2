# -*- coding: utf-8 -*-
import click
import logging
import cv2
from pathlib import Path


@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_video, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        intermediate data ready to be classified (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    count = 0
    cap = cv2.VideoCapture(input_video)
    success, image = cap.read()
    logger.info('Starting getting frames from video...')

    while success:
        cap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        success, image = cap.read()
        logger.debug('Read a new frame: {}'.format(success))
        if not success:
            break

        cv2.imwrite(str(output_filepath) + "frame{}.jpg".format(count), image)
        count = count + 1

    logger.info('Done extracting frames from the video.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()

