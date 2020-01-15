# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import itertools
from typing import Any, Iterator, List, Union

import numpy as np
import pycocotools.mask as mask_utils
import torch


def polygon_area(x, y):
    # Using the shoelace formula
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygons_to_bitmask(polygons: List[np.ndarray], height: int,
                        width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.bool)


def rasterize_polygons_within_box(polygons: List[np.ndarray], box: np.ndarray,
                                  mask_size: int) -> torch.Tensor:
    """
    Rasterize the polygons into a mask image and
    crop the mask content in the given box.
    The cropped mask is resized to (mask_size, mask_size).

    This function is used when generating training targets for mask head in Mask R-CNN.
    Given original ground-truth masks for an image, new ground-truth mask
    training targets in the size of `mask_size x mask_size`
    must be provided for each predicted box. This function will be called to
    produce such targets.

    Args:
        polygons (list[ndarray[float]]): a list of polygons, which represents an instance.
        box: 4-element numpy array
        mask_size (int):

    Returns:
        Tensor: BoolTensor of shape (mask_size, mask_size)
    """
    # 1. Shift the polygons w.r.t the boxes
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    # 2. Rescale the polygons to the new box size
    ratio_h = mask_size / max(h, 0.1)
    ratio_w = mask_size / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    mask = polygons_to_bitmask(polygons, mask_size, mask_size)
    mask = torch.from_numpy(mask)
    return mask


class BitMasks:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
        """
        device = tensor.device if isinstance(
            tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.bool, device=device)
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    def to(self, device: str) -> "BitMasks":
        return BitMasks(self.tensor.to(device))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(self, item: Union[int, slice,
                                      torch.BoolTensor]) -> "BitMasks":
        """
        Returns:
            BitMasks: Create a new :class:`BitMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BitMasks(self.tensor[item].view(1, -1))
        m = self.tensor[item]
        assert m.dim(
        ) == 3, "Indexing on BitMasks with {} returns a tensor with shape {}!".format(
            item, m.shape)
        return BitMasks(m)

    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)

    @staticmethod
    def from_polygon_masks(polygon_masks: Union["PolygonMasks",
                                                List[List[np.ndarray]]],
                           height: int, width: int) -> "BitMasks":
        """
        Args:
            polygon_masks (list[list[ndarray]] or PolygonMasks)
            height, width (int)
        """
        if isinstance(polygon_masks, PolygonMasks):
            polygon_masks = polygon_masks.polygons
        masks = [polygons_to_bitmask(p, height, width) for p in polygon_masks]
        return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))

    def crop_and_resize(self, boxes: torch.Tensor,
                        mask_size: int) -> torch.Tensor:
        raise NotImplementedError

    def get_bounding_boxes(self) -> None:
        # not needed now
        raise NotImplementedError

    @staticmethod
    def cat(bitmasks_list: List["BitMasks"]) -> "BitMasks":
        raise NotImplementedError


class PolygonMasks:
    """
    This class stores the segmentation masks for all objects in one image, in the form of polygons.

    Attributes:
        polygons: list[list[ndarray]]. Each ndarray is a float64 vector representing a polygon.
    """

    def __init__(self,
                 polygons: List[List[Union[torch.Tensor, np.ndarray]]],
                 size=None):
        """
        Arguments:
            polygons (list[list[Tensor[float]]]): The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                instance, and the third level to the polygon coordinates.
                The third level Tensor should have the format of
                torch.Tensor([x0, y0, x1, y1, ..., xn, yn]) (n >= 3).
        """
        self.size = size
        assert isinstance(polygons, list), (
            "Cannot create PolygonMasks: Expect a list of list of polygons per image. "
            "Got '{}' instead.".format(type(polygons)))

        def _make_array(t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
            # Use float64 for higher precision, because why not?
            # Always put polygons on CPU (self.to is a no-op) since they
            # are supposed to be small tensors.
            # May need to change this assumption if GPU placement becomes useful
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            return np.asarray(t).astype("float64")

        def process_polygons(
            polygons_per_instance: List[Union[torch.Tensor, np.ndarray]]
        ) -> List[torch.Tensor]:
            assert isinstance(polygons_per_instance, list), (
                "Cannot create polygons: Expect a list of polygons per instance. "
                "Got '{}' instead.".format(type(polygons_per_instance)))
            # transform the polygon to a tensor
            polygons_per_instance = [
                _make_array(p) for p in polygons_per_instance
            ]
            for polygon in polygons_per_instance:
                assert len(polygon) % 2 == 0 and len(polygon) >= 6
            return polygons_per_instance

        self.polygons: List[List[torch.Tensor]] = [
            process_polygons(polygons_per_instance)
            for polygons_per_instance in polygons
        ]

    def to(self, *args: Any, **kwargs: Any) -> "PolygonMasks":
        return self

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor:
                a BoolTensor which represents whether each mask is empty (False) or not (True).
        """
        keep = [1 if len(polygon) > 0 else 0 for polygon in self.polygons]
        return torch.as_tensor(keep, dtype=torch.bool)

    def __getitem__(
        self, item: Union[int, slice, List[int], torch.BoolTensor]
    ) -> "PolygonMasks":
        """
        Support indexing over the instances and return a `PolygonMasks` object.
        `item` can be:

        1. An integer. It will return an object with only one instance.
        2. A slice. It will return an object with the selected instances.
        3. A list[int]. It will return an object with the selected instances,
           correpsonding to the indices in the list.
        4. A vector mask of type BoolTensor, whose length is num_instances.
           It will return an object with the instances whose mask is nonzero.
        """
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        elif isinstance(item, list):
            selected_polygons = [self.polygons[i] for i in item]
        elif isinstance(item, torch.Tensor):
            # Polygons is a list, so we have to move the indices back to CPU.
            if item.dtype == torch.bool:
                assert item.dim() == 1, item.shape
                item = item.nonzero().squeeze(1).cpu().numpy().tolist()
            elif item.dtype in [torch.int32, torch.int64]:
                item = item.cpu().numpy().tolist()
            else:
                raise ValueError(
                    "Unsupported tensor dtype={} for indexing!".format(
                        item.dtype))
            selected_polygons = [self.polygons[i] for i in item]
        return PolygonMasks(selected_polygons)

    def __iter__(self) -> Iterator[List[torch.Tensor]]:
        """
        Yields:
            list[ndarray]: the polygons for one instance.
            Each Tensor is a float64 vector representing a polygon.
        """
        return iter(self.polygons)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.polygons))
        return s

    def __len__(self) -> int:
        return len(self.polygons)

    def crop_and_resize(self, boxes: torch.Tensor,
                        mask_size: int) -> torch.Tensor:
        """
        Crop each mask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.

        Returns:
            Tensor: A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
        """
        assert len(boxes) == len(self), "{} != {}".format(
            len(boxes), len(self))

        device = boxes.device
        # Put boxes on the CPU, as the polygon representation is not efficient GPU-wise
        # (several small tensors for representing a single instance mask)
        boxes = boxes.to(torch.device("cpu"))

        results = [
            rasterize_polygons_within_box(poly, box.numpy(), mask_size)
            for poly, box in zip(self.polygons, boxes)
        ]
        """
        poly: list[list[float]], the polygons for one instance
        box: a tensor of shape (4,)
        """
        if len(results) == 0:
            return torch.empty(
                0, mask_size, mask_size, dtype=torch.bool, device=device)
        return torch.stack(results, dim=0).to(device=device)

    def area(self):
        """
        Computes area of the mask.
        Only works with Polygons, using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

        Returns:
            Tensor: a vector, area for each instance
        """

        area = []
        for polygons_per_instance in self.polygons:
            area_per_instance = 0
            for p in polygons_per_instance:
                area_per_instance += polygon_area(p[0::2], p[1::2])
            area.append(area_per_instance)

        return torch.tensor(area)

    @staticmethod
    def cat(polymasks_list: List["PolygonMasks"]) -> "PolygonMasks":
        """
        Concatenates a list of PolygonMasks into a single PolygonMasks

        Arguments:
            polymasks_list (list[PolygonMasks])

        Returns:
            PolygonMasks: the concatenated PolygonMasks
        """
        assert isinstance(polymasks_list, (list, tuple))
        assert len(polymasks_list) > 0
        assert all(
            isinstance(polymask, PolygonMasks) for polymask in polymasks_list)

        cat_polymasks = type(polymasks_list[0])(
            list(
                itertools.chain.from_iterable(pm.polygons
                                              for pm in polymasks_list)))
        return cat_polymasks

    def resize(self, new_size):
        ratio_h = new_size[0] / self.size[0]
        ratio_w = new_size[1] / self.size[1]
        scaled_polygons = []
        for polygon in self.polygons:
            scaled_polygon = []
            for p in polygon:
                p = p.copy()
                p[0::2] *= ratio_w
                p[1::2] *= ratio_h
                scaled_polygon.append(p)
            scaled_polygons.append(scaled_polygon)
        self.polygons = scaled_polygons
        self.size = new_size

    def flip(self, direction):
        if direction == 'horizontal':
            flipped_polygons = []
            for poly in self.polygons:
                flip_poly = []
                for p in poly:
                    p = p.copy()
                    p[0::2] = self.size[1] - p[0::2]
                    flip_poly.append(p)
                flipped_polygons.append(flip_poly)
            self.polygons = flipped_polygons
        else:
            assert NotImplementedError
