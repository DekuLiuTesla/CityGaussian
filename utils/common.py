import os
import queue
import threading
import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor
import torch
import numpy as np
import cv2


class AsyncTensorSaver:
    def __init__(self, maxsize: int = 16):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._save_tensor_from_queue)
        self.thread.start()

    def _save_tensor_from_queue(self):
        while True:
            t = self.queue.get()
            if t is None:
                break
            tensor, path = t
            self._sync_save_tensor_to_file(tensor, path)

    def save(self, tensor, path):
        self.queue.put((tensor.cpu(), path))

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def _sync_save_tensor_to_file(tensor, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(tensor, path + ".tmp")
        os.rename(path + ".tmp", path)


class AsyncNDArraySaver:
    def __init__(self, maxsize: int = 16):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._save_ndarray_from_queue)
        self.thread.start()

    def _save_ndarray_from_queue(self):
        while True:
            t = self.queue.get()
            if t is None:
                break
            ndarray, path = t
            self._sync_save_ndarray_to_file(ndarray, path)

    def save(self, ndarray, path):
        self.queue.put((ndarray, path))

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def _sync_save_ndarray_to_file(ndarray, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, ndarray)


class AsyncImageSaver:
    def __init__(self, maxsize=16, is_rgb: bool = False):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._save_image_from_queue)
        self.thread.start()

        self.is_rgb = is_rgb

    def save(self, image, path, processor=None):
        self.queue.put((image, path, processor))

    def _save_image_from_queue(self):
        while True:
            i = self.queue.get()
            if i is None:
                break
            image, path, func = i
            self._sync_save_image(image, path, func, self.is_rgb)

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def _sync_save_image(image, path, func, is_rgb):
        dot_index = path.rfind(".")
        ext = path[dot_index:]

        tmp_path = f"{path[:dot_index]}.tmp{ext}"

        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

        if func is not None:
            image = func(image)
        if image.shape[-1] == 3 and is_rgb is True:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_path, image)
        os.rename(tmp_path, path)


class AsyncImageReader:
    def __init__(self, image_list: list, maxsize=16):
        self.image_list = image_list.copy()
        self.queue = queue.Queue(maxsize=maxsize)

        self.tpe = ThreadPoolExecutor(max_workers=4)

        self.is_finished = False

        self.thread = threading.Thread(target=self._read_images)
        self.thread.start()


    def get(self):
        return self.queue.get()

    def _read_images(self):
        results = self.tpe.map(self._read_image_to_queue, self.image_list)
        try:
            for _ in results:
                pass
        except concurrent.futures.CancelledError:
            pass
        self.tpe.shutdown()
        self.is_finished = True

    def _read_image_to_queue(self, image_path):
        self.queue.put((image_path, cv2.imread(image_path)))

    def stop(self):
        while self.is_finished is False:
            try:
                self.queue.get(block=False)
            except:
                pass
            self.tpe.shutdown(wait=False, cancel_futures=True)


def find_files(dir: str, extensions: list[str], as_relative_path: bool = True) -> list[str]:
    from glob import glob

    image_list = []
    for ext in extensions:
        image_list += list(glob(os.path.join(dir, "**/*.{}".format(ext)), recursive=True))
    image_list.sort()

    dir_len = len(dir)
    if as_relative_path is True:
        image_list = [i[dir_len:].lstrip("/") for i in image_list]

    return image_list
