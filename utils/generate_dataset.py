"""This script is used to transform a folder with raw recordings
into a folder ready to use for a DatasetFolder class.
NOTE: Should be changed to work on folder with class folders"""
# pylint: disable=import-error,redefined-outer-name

import os
import json

import yaml

from torchaudio import load, save


def get_file_class(filename, task):
    """Determine file class for task defined in config.

    Args:
        filename (str): Name of file.
        task (str): Name of classification task.

    Returns:
        str: class name of the file.
    """
    split_idx = (
        0
        if task.lower() in ("user", "u")
        else 1
        if task.lower() in ("material", "m")
        else 2
    )
    file_class = filename.split("_")[split_idx]
    if split_idx == 2:
        file_class = int(file_class)
        # pylint: disable=invalid-name
        file_class = (
            "slow" if file_class < 3 else "fast" if file_class > 4 else "medium"
        )
    return file_class


def read_annotation(filename, annotation_folder):
    """Get starting and ending sample of the file.

    Args:
        filename (str): Name of file.
        annotation_folder (str): Name of folder containing annotations.

    Returns:
        tuple(str, int, int): Tuple of audio file name, start and end sample of annotation.
    """
    annotation_file = os.path.join(annotation_folder, filename)
    with open(annotation_file, "r", encoding="utf-8") as file:
        annotation = json.load(file)
    annotated_events = annotation["audio_annotations"]
    return (
        annotated_events,
        annotation["audio_file"],
    )


if __name__ == "__main__":
    # load config
    with open("C:\\Users\\sokol\\OneDrive\\Pulpit\\SEM_8\\krypto\\lab1\\audio_ml\\config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config = config["data"]
    for root, _, files in os.walk(config["annotation_folder"]):
        for file in files:
            # get & check file extension
            filename, extension = os.path.splitext(file)
            if extension.lower() != ".json":
                continue
            # read annotation file
            try:
                events, audio_file = read_annotation(file, config["annotation_folder"])
            except KeyError:
                continue
            # load audio file
            try:
                signal, sr = load(os.path.join(config["data_folder"], audio_file))
            except RuntimeError:
                continue
            # determine class
            file_class = int(file.split("_")[1][:2])
            # prepare target directory
            path = os.path.join(config["target_folder"], str(file_class))
            os.makedirs(path, exist_ok=True)
            # get the relevant signal part
            start_sample = events["1"]["sample"]
            end_sample = events["2"]["sample"]
            signal_part = signal[:, start_sample:end_sample]
            # calculate desired slice length
            slice_len = int(sr * config["slice_length"])
            overlap_offset = int(slice_len * (1 - config["overlap"]))
            # pylint: disable=invalid-name
            slice_no = 0
            # slice it up!
            # pylint: disable=invalid-name
            while signal_part.shape[1] > slice_len + slice_no * overlap_offset:
                # select file class as puncture or tissue
                slice_begin, slice_end = (
                    slice_no * overlap_offset,
                    slice_len + slice_no * overlap_offset,
                )
                # create directory for slices
                path = os.path.join(
                    config["target_folder"], str(file_class), audio_file
                )
                os.makedirs(path, exist_ok=True)
                # take last slice_len of spectrogram and save it
                current_slice = signal_part[
                    :,
                    slice_no * overlap_offset : slice_len + slice_no * overlap_offset,
                ]
                save(os.path.join(path, f"slice{slice_no}.wav"), current_slice, sr)
                slice_no += 1
