import time
from datetime import datetime
import json
import shutil
import os
import pickle
import numpy as np
import pandas as pd


def to_datetime(date_time, unit="ms"):
    if isinstance(date_time, np.int64) or isinstance(date_time, float) or isinstance(date_time, int):
        date_time = pd.to_datetime(date_time, unit=unit)
    try:
        for fdt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S'):
            try:
                return pd.to_datetime(
                    [date_time],
                    format=fdt
                )[0]
            except Exception:
                pass
    except KeyboardInterrupt as e:
        raise e
    except Exception:
        raise ValueError(f'No valid date format found for {date_time}')


def convertVarArgToString(*messages):
    full_message = ""
    for index in range(len(messages)):
        full_message += str(messages[index])
        if index < len(messages) - 1:
            full_message += " "
    return full_message


class Logger:
    log_path = "./cache/" + str(int(time.time())) + ".log"

    @staticmethod
    def set_log_path(log_path):
        Logger.log_path = "./cache/" + log_path + ".log"

    @staticmethod
    def log_m(*messages, path=""):
        Logger.logToScreen(*messages)
        Logger.logToFile(*messages, path=path)

    @staticmethod
    def log_e(*errors, path=""):
        Logger.log_m("Error :", *errors, path=path)

    @staticmethod
    def logToScreen(*messages):
        print(convertVarArgToString(*messages))

    @staticmethod
    def logToFile(*messages, path=""):
        if path == "":
            path = Logger.log_path
        try:
            with open(path, "a") as log_file:
                log_file.write(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                    " > " +
                    convertVarArgToString(*messages) +
                    "\n"
                )
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(e)


def write_json_file(file_path, data):
    file = open(file_path, "w")
    file.write(
        json.dumps(
            data,
            sort_keys=True,
            indent=4
        )
    )


def read_json_file(file_path, default=None):
    if is_file_exist(file_path=file_path):
        file = open(file_path, "r")
        return json.loads(file.read())
    else:
        return default


def is_file_exist(file_path):
    try:
        open(file_path, "r")
        return True
    except KeyboardInterrupt as e:
        raise e
    except FileNotFoundError:
        return False


def is_folder_exist(folder_path):
    return os.path.isdir(folder_path)


def write_pickle_file(file_path, data):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def read_pickle_file(file_path, default=None):
    if is_file_exist(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
    else:
        return default


def copy_and_replace_file(source, destination, verbose=1):
    try:
        shutil.copy2(source, destination)  # copies src file to dst path (overwrites if exists)
        if verbose >= 1:
            print(f"File copied from {source} to {destination}")
    except KeyboardInterrupt as e:
        raise e
    except FileNotFoundError:
        print(f"{source} not found.")
    except Exception as e:
        print(str(e))
