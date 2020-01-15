# -*- coding: utf-8 -*-
import asyncio
import click
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Union
from imageai.Prediction.Custom import CustomImagePrediction
from helpers.utils import print_progress, gather_dict
from functools import wraps


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


async def predict_image(
    image_name: Union[str, np.ndarray], model: CustomImagePrediction
) -> dict:
    """
    Predicts a given image with the supplied prediction model.
    :param image_name:
    :param model:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Predicting the {} image".format(image_name))

    predictions, probabilities = model.predictImage(
        (PROJECT_DIR / image_name).as_posix(), result_count=2
    )

    representation = {}
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        representation[eachPrediction] = "{0:.2f}%".format(float(eachProbability))

    return representation


async def predict_video(video_path: str, model: CustomImagePrediction):
    """
    Splits a video file and predicts for each 1s frame image with the
    supplied prediction model.
    :param video_path:
    :param model:
    """
    cap = cv2.VideoCapture(video_path)
    VIDEO_DURATION_IN_SECONDS = (
        int(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)) + 1
    )
    # remove episode3_results images
    # os.system('rm -rf episode3_results')
    logger = logging.getLogger(__name__)
    logger.info("Gathering frames from the video...")

    counter = 0
    tasks = {}
    # read all frames and run predictions on them
    while cap.isOpened():
        # set position to only read full seconds
        cap.set(cv2.CAP_PROP_POS_MSEC, (counter * 1000))
        ret, frame = cap.read()

        if not ret:
            break

        # store the image and use that one for predicting
        image_name = "episode3_results/ep3_frame{}.jpg".format(counter)
        cv2.imwrite(image_name, frame)

        tasks[image_name] = asyncio.ensure_future(predict_image(image_name, model))
        counter += 1
        print_progress(counter / VIDEO_DURATION_IN_SECONDS)

    cap.release()
    cv2.destroyAllWindows()

    logger.info("Getting predictions foreach frame from the model...")
    results = await gather_dict(tasks)

    logger.info("Writing predictions on images....")

    for image_path in results.keys():
        logger.info("Writing prediction for {}".format(image_path))
        img = cv2.imread(image_path)

        results_string = ""
        for res in sorted(results[image_path].keys()):
            results_string += " " + res + " " + results[image_path][res]

        cv2.putText(
            img,
            results_string,
            (130, 25),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(image_path, img)

    logger.info("Done....")


@click.command()
@coro
@click.option(
    "--file_type",
    "-t",
    required=True,
    type=str,
    default="video",
    help="Input file type",
)
@click.option(
    "--files",
    "-f",
    required=True,
    type=str,
    help="File path. Comma separated images accepted if type image",
)
async def main(file_type, files):
    logger = logging.getLogger(__name__)
    logger.info("Started evaluating input files.")

    model = CustomImagePrediction()
    model.setModelTypeAsResNet()

    model.setModelPath((PROJECT_DIR / "data/images/models/model_ex-001_acc-0.671875.h5").as_posix())
    model.setJsonPath((PROJECT_DIR / "data/images/json/model_class.json").as_posix())
    model.loadModel(num_objects=2)  # number of objects on your trained model

    if file_type == "image":
        for image in files.split(","):
            logger.info(await predict_image(image_name=image, model=model))
    else:
        await predict_video(video_path=files, model=model)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    PROJECT_DIR = Path(__file__).resolve().parents[1]

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
