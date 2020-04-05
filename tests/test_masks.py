import numpy as np
import pytest
import torch

from mmdet.core import BitmapMasks, PolygonMasks


def dummy_raw_bitmap_masks(size):
    """
    Args:
        size (tuple): expected shape of dummy masks, (H, W) or (N, H, W)

    Return:
        ndarray: dummy mask
    """
    return np.random.randint(0, 2, size, dtype=np.uint8)


def dummy_raw_polygon_masks(size):
    """
    Args:
        size (tuple): expected shape of dummy masks, (N, H, W)

    Return:
        list[list[ndarray]]: dummy mask
    """
    num_obj, heigt, width = size
    polygons = []
    for _ in range(num_obj):
        num_points = np.random.randint(5) * 2 + 6
        polygons.append([np.random.uniform(0, min(heigt, width), num_points)])
    return polygons


def dummy_bboxes(num, max_height, max_width):
    x1y1 = np.random.randint(0, min(max_height // 2, max_width // 2), (num, 2))
    wh = np.random.randint(0, min(max_height // 2, max_width // 2), (num, 2))
    x2y2 = x1y1 + wh
    return np.concatenate([x1y1, x2y2], axis=1).squeeze()


def test_bitmap_mask_init():
    # init with empty ndarray masks
    raw_masks = np.empty((0, 28, 28), dtype=np.uint8)
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    assert len(bitmap_masks) == 0
    assert bitmap_masks.height == 28
    assert bitmap_masks.width == 28

    # init with empty list masks
    raw_masks = []
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    assert len(bitmap_masks) == 0
    assert bitmap_masks.height == 28
    assert bitmap_masks.width == 28

    # init with ndarray masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    assert len(bitmap_masks) == 3
    assert bitmap_masks.height == 28
    assert bitmap_masks.width == 28

    # init with list masks contain 3 instances
    raw_masks = [dummy_raw_bitmap_masks((28, 28)) for _ in range(3)]
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    assert len(bitmap_masks) == 3
    assert bitmap_masks.height == 28
    assert bitmap_masks.width == 28

    # init with raw masks of unsupported type
    with pytest.raises(AssertionError):
        raw_masks = [[dummy_raw_bitmap_masks((28, 28))]]
        BitmapMasks(raw_masks, 28, 28)


def test_bitmap_mask_rescale():
    # rescale with empty bitmap masks
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    rescaled_masks = bitmap_masks.rescale((56, 72))
    assert len(rescaled_masks) == 0
    assert rescaled_masks.height == 56
    assert rescaled_masks.width == 56

    # rescale with bitmap masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    rescaled_masks = bitmap_masks.rescale((56, 72))
    assert len(rescaled_masks) == 3
    assert rescaled_masks.height == 56
    assert rescaled_masks.width == 56


def test_bitmap_mask_resize():
    # resize with empty bitmap masks
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    resized_masks = bitmap_masks.resize((56, 72))
    assert len(resized_masks) == 0
    assert resized_masks.height == 56
    assert resized_masks.width == 72

    # resize with bitmap masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    resized_masks = bitmap_masks.resize((56, 72))
    assert len(resized_masks) == 3
    assert resized_masks.height == 56
    assert resized_masks.width == 72


def test_bitmap_mask_flip():
    # flip with empty bitmap masks
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    flipped_masks = bitmap_masks.flip(flip_direction='horizontal')
    assert len(flipped_masks) == 0
    assert flipped_masks.height == 28
    assert flipped_masks.width == 28

    # horizontally flip with bitmap masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    flipped_masks = bitmap_masks.flip(flip_direction='horizontal')
    flipped_flipped_masks = flipped_masks.flip(flip_direction='horizontal')
    assert flipped_masks.masks.shape == (3, 28, 28)
    assert (bitmap_masks.masks == flipped_flipped_masks.masks).all()
    assert (flipped_masks.masks == raw_masks[:, :, ::-1]).all()

    # vertically flip with bitmap masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    flipped_masks = bitmap_masks.flip(flip_direction='vertical')
    flipped_flipped_masks = flipped_masks.flip(flip_direction='vertical')
    assert len(flipped_masks) == 3
    assert flipped_masks.height == 28
    assert flipped_masks.width == 28
    assert (bitmap_masks.masks == flipped_flipped_masks.masks).all()
    assert (flipped_masks.masks == raw_masks[:, ::-1, :]).all()


def test_bitmap_mask_pad():
    # pad with empty bitmap masks
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    padded_masks = bitmap_masks.pad((56, 56))
    assert len(padded_masks) == 0
    assert padded_masks.height == 56
    assert padded_masks.width == 56

    # pad with bitmap masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    padded_masks = bitmap_masks.pad((56, 56))
    assert len(padded_masks) == 3
    assert padded_masks.height == 56
    assert padded_masks.width == 56
    assert (padded_masks.masks[:, 28:, 28:] == 0).all()


def test_bitmap_mask_crop():
    # crop with empty bitmap masks
    dummy_bbox = np.array([0, 10, 10, 27], dtype=np.int)
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    cropped_masks = bitmap_masks.crop(dummy_bbox)
    assert len(cropped_masks) == 0
    assert cropped_masks.height == 18
    assert cropped_masks.width == 11

    # crop with bitmap masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    cropped_masks = bitmap_masks.crop(dummy_bbox)
    assert len(cropped_masks) == 3
    assert cropped_masks.height == 18
    assert cropped_masks.width == 11
    x1, y1, x2, y2 = dummy_bbox
    assert (cropped_masks.masks == raw_masks[:, y1:y2 + 1, x1:x2 + 1]).all()

    # crop with invalid bbox
    with pytest.raises(AssertionError):
        dummy_bbox = dummy_bboxes(2, 28, 28)
        bitmap_masks.crop(dummy_bbox)


def test_bitmap_mask_crop_and_resize():
    dummy_bbox = dummy_bboxes(5, 28, 28)
    inds = np.random.randint(0, 3, (5, ))

    # crop and resize with empty bitmap masks
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    cropped_resized_masks = bitmap_masks.crop_and_resize(
        dummy_bbox, (56, 56), inds)
    assert len(cropped_resized_masks) == 0
    assert cropped_resized_masks.height == 56
    assert cropped_resized_masks.width == 56

    # crop and resize with bitmap masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    cropped_resized_masks = bitmap_masks.crop_and_resize(
        dummy_bbox, (56, 56), inds)
    assert len(cropped_resized_masks) == 5
    assert cropped_resized_masks.height == 56
    assert cropped_resized_masks.width == 56


def test_bitmap_mask_expand():
    # expand with empty bitmap masks
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    expanded_masks = bitmap_masks.expand(56, 56, 12, 14)
    assert len(expanded_masks) == 0
    assert expanded_masks.height == 56
    assert expanded_masks.width == 56

    # expand with bitmap masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    expanded_masks = bitmap_masks.expand(56, 56, 12, 14)
    assert len(expanded_masks) == 3
    assert expanded_masks.height == 56
    assert expanded_masks.width == 56
    assert (expanded_masks.masks[:, :12, :14] == 0).all()
    assert (expanded_masks.masks[:, 12 + 28:, 14 + 28:] == 0).all()


def test_bitmap_mask_area():
    # area of empty bitmap mask
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    assert bitmap_masks.area().sum() == 0

    # area of bitmap masks contain 3 instances
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    areas = bitmap_masks.area()
    assert len(areas) == 3
    assert (areas == raw_masks.sum((1, 2))).all()


def test_bitmap_mask_to_ndarray():
    # empty bitmap masks to ndarray
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    ndarray_masks = bitmap_masks.to_ndarray()
    assert isinstance(ndarray_masks, np.ndarray)
    assert ndarray_masks.shape == (0, 28, 28)

    # bitmap masks contain 3 instances to ndarray
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    ndarray_masks = bitmap_masks.to_ndarray()
    assert isinstance(ndarray_masks, np.ndarray)
    assert ndarray_masks.shape == (3, 28, 28)
    assert (ndarray_masks == raw_masks).all()


def test_bitmap_mask_to_tensor():
    # empty bitmap masks to tensor
    raw_masks = dummy_raw_bitmap_masks((0, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    tensor_masks = bitmap_masks.to_tensor(dtype=torch.uint8, device='cpu')
    assert isinstance(tensor_masks, torch.Tensor)
    assert tensor_masks.shape == (0, 28, 28)

    # bitmap masks contain 3 instances to tensor
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    tensor_masks = bitmap_masks.to_tensor(dtype=torch.uint8, device='cpu')
    assert isinstance(tensor_masks, torch.Tensor)
    assert tensor_masks.shape == (3, 28, 28)
    assert (tensor_masks.numpy() == raw_masks).all()


def test_bitmap_mask_index():
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    assert (bitmap_masks[0].masks == raw_masks[0]).all()
    assert (bitmap_masks[range(2)].masks == raw_masks[range(2)]).all()


def test_bitmap_mask_iter():
    raw_masks = dummy_raw_bitmap_masks((3, 28, 28))
    bitmap_masks = BitmapMasks(raw_masks, 28, 28)
    for i, bitmap_mask in enumerate(bitmap_masks):
        assert bitmap_mask.shape == (28, 28)
        assert (bitmap_mask == raw_masks[i]).all()


def test_polygon_mask_init():
    # init with empty masks
    raw_masks = []
    polygon_masks = BitmapMasks(raw_masks, 28, 28)
    assert len(polygon_masks) == 0
    assert polygon_masks.height == 28
    assert polygon_masks.width == 28

    # init with masks contain 3 instances
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    assert isinstance(polygon_masks.masks, list)
    assert isinstance(polygon_masks.masks[0], list)
    assert isinstance(polygon_masks.masks[0][0], np.ndarray)
    assert len(polygon_masks) == 3
    assert polygon_masks.height == 28
    assert polygon_masks.width == 28
    assert polygon_masks.to_ndarray().shape == (3, 28, 28)

    # init with raw masks of unsupported type
    with pytest.raises(AssertionError):
        raw_masks = [[[]]]
        PolygonMasks(raw_masks, 28, 28)

        raw_masks = [dummy_raw_polygon_masks((3, 28, 28))]
        PolygonMasks(raw_masks, 28, 28)


def test_polygon_mask_rescale():
    # rescale with empty polygon masks
    raw_masks = dummy_raw_polygon_masks((0, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    rescaled_masks = polygon_masks.rescale((56, 72))
    assert len(rescaled_masks) == 0
    assert rescaled_masks.height == 56
    assert rescaled_masks.width == 56
    assert rescaled_masks.to_ndarray().shape == (0, 56, 56)

    # rescale with polygon masks contain 3 instances
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    rescaled_masks = polygon_masks.rescale((56, 72))
    assert len(rescaled_masks) == 3
    assert rescaled_masks.height == 56
    assert rescaled_masks.width == 56
    assert rescaled_masks.to_ndarray().shape == (3, 56, 56)


def test_polygon_mask_resize():
    # resize with empty polygon masks
    raw_masks = dummy_raw_polygon_masks((0, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    resized_masks = polygon_masks.resize((56, 72))
    assert len(resized_masks) == 0
    assert resized_masks.height == 56
    assert resized_masks.width == 72
    assert resized_masks.to_ndarray().shape == (0, 56, 72)

    # resize with polygon masks contain 3 instances
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    resized_masks = polygon_masks.resize((56, 72))
    assert len(resized_masks) == 3
    assert resized_masks.height == 56
    assert resized_masks.width == 72
    assert resized_masks.to_ndarray().shape == (3, 56, 72)


def test_polygon_mask_flip():
    # flip with empty polygon masks
    raw_masks = dummy_raw_polygon_masks((0, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    flipped_masks = polygon_masks.flip(flip_direction='horizontal')
    assert len(flipped_masks) == 0
    assert flipped_masks.height == 28
    assert flipped_masks.width == 28
    assert flipped_masks.to_ndarray().shape == (0, 28, 28)

    # TODO: fixed flip correctness checking after v2.0_coord is merged
    # horizontally flip with polygon masks contain 3 instances
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    flipped_masks = polygon_masks.flip(flip_direction='horizontal')
    flipped_flipped_masks = flipped_masks.flip(flip_direction='horizontal')
    assert len(flipped_masks) == 3
    assert flipped_masks.height == 28
    assert flipped_masks.width == 28
    assert flipped_masks.to_ndarray().shape == (3, 28, 28)
    assert (polygon_masks.to_ndarray() == flipped_flipped_masks.to_ndarray()
            ).all()

    # vertically flip with polygon masks contain 3 instances
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    flipped_masks = polygon_masks.flip(flip_direction='vertical')
    flipped_flipped_masks = flipped_masks.flip(flip_direction='vertical')
    assert len(flipped_masks) == 3
    assert flipped_masks.height == 28
    assert flipped_masks.width == 28
    assert flipped_masks.to_ndarray().shape == (3, 28, 28)
    assert (polygon_masks.to_ndarray() == flipped_flipped_masks.to_ndarray()
            ).all()


def test_polygon_mask_crop():
    dummy_bbox = np.array([0, 10, 10, 27], dtype=np.int)
    # crop with empty polygon masks
    raw_masks = dummy_raw_polygon_masks((0, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    cropped_masks = polygon_masks.crop(dummy_bbox)
    assert len(cropped_masks) == 0
    assert cropped_masks.height == 18
    assert cropped_masks.width == 11
    assert cropped_masks.to_ndarray().shape == (0, 18, 11)

    # crop with polygon masks contain 3 instances
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    cropped_masks = polygon_masks.crop(dummy_bbox)
    assert len(cropped_masks) == 3
    assert cropped_masks.height == 18
    assert cropped_masks.width == 11
    assert cropped_masks.to_ndarray().shape == (3, 18, 11)

    # crop with invalid bbox
    with pytest.raises(AssertionError):
        dummy_bbox = dummy_bboxes(2, 28, 28)
        polygon_masks.crop(dummy_bbox)


def test_polygon_mask_pad():
    # pad with empty polygon masks
    raw_masks = dummy_raw_polygon_masks((0, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    padded_masks = polygon_masks.pad((56, 56))
    assert len(padded_masks) == 0
    assert padded_masks.height == 56
    assert padded_masks.width == 56
    assert padded_masks.to_ndarray().shape == (0, 56, 56)

    # pad with polygon masks contain 3 instances
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    padded_masks = polygon_masks.pad((56, 56))
    assert len(padded_masks) == 3
    assert padded_masks.height == 56
    assert padded_masks.width == 56
    assert padded_masks.to_ndarray().shape == (3, 56, 56)
    assert (padded_masks.to_ndarray()[:, 28:, 28:] == 0).all()


def test_polygon_mask_expand():
    with pytest.raises(NotImplementedError):
        raw_masks = dummy_raw_polygon_masks((0, 28, 28))
        polygon_masks = PolygonMasks(raw_masks, 28, 28)
        polygon_masks.expand(56, 56, 10, 17)


def test_polygon_mask_crop_and_resize():
    dummy_bbox = dummy_bboxes(5, 28, 28)
    inds = np.random.randint(0, 3, (5, ))

    # crop and resize with empty polygon masks
    raw_masks = dummy_raw_polygon_masks((0, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    cropped_resized_masks = polygon_masks.crop_and_resize(
        dummy_bbox, (56, 56), inds)
    assert len(cropped_resized_masks) == 0
    assert cropped_resized_masks.height == 56
    assert cropped_resized_masks.width == 56
    assert cropped_resized_masks.to_ndarray().shape == (0, 56, 56)

    # crop and resize with polygon masks contain 3 instances
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    cropped_resized_masks = polygon_masks.crop_and_resize(
        dummy_bbox, (56, 56), inds)
    assert len(cropped_resized_masks) == 5
    assert cropped_resized_masks.height == 56
    assert cropped_resized_masks.width == 56
    assert cropped_resized_masks.to_ndarray().shape == (5, 56, 56)


def test_polygon_mask_area():
    # area of empty polygon masks
    raw_masks = dummy_raw_polygon_masks((0, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    assert polygon_masks.area().sum() == 0

    # area of polygon masks contain 1 instance
    # here we hack a case that the gap between the area of bitmap and polygon
    # is minor
    raw_masks = [[np.array([1, 1, 5, 1, 3, 4])]]
    polygon_masks = PolygonMasks(raw_masks, 6, 6)
    polygon_area = polygon_masks.area()
    bitmap_area = polygon_masks.to_bitmap().area()
    assert len(polygon_area) == 1
    assert np.isclose(polygon_area, bitmap_area).all()


def test_polygon_mask_to_bitmap():
    # polygon masks contain 3 instances to bitmap
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    bitmap_masks = polygon_masks.to_bitmap()
    assert (polygon_masks.to_ndarray() == bitmap_masks.to_ndarray()).all()


def test_polygon_mask_to_ndarray():
    # empty polygon masks to ndarray
    raw_masks = dummy_raw_polygon_masks((0, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    ndarray_masks = polygon_masks.to_ndarray()
    assert isinstance(ndarray_masks, np.ndarray)
    assert ndarray_masks.shape == (0, 28, 28)

    # polygon masks contain 3 instances to ndarray
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    ndarray_masks = polygon_masks.to_ndarray()
    assert isinstance(ndarray_masks, np.ndarray)
    assert ndarray_masks.shape == (3, 28, 28)


def test_polygon_to_tensor():
    # empty polygon masks to tensor
    raw_masks = dummy_raw_polygon_masks((0, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    tensor_masks = polygon_masks.to_tensor(dtype=torch.uint8, device='cpu')
    assert isinstance(tensor_masks, torch.Tensor)
    assert tensor_masks.shape == (0, 28, 28)

    # polygon masks contain 3 instances to tensor
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    tensor_masks = polygon_masks.to_tensor(dtype=torch.uint8, device='cpu')
    assert isinstance(tensor_masks, torch.Tensor)
    assert tensor_masks.shape == (3, 28, 28)
    assert (tensor_masks.numpy() == polygon_masks.to_ndarray()).all()


def test_polygon_mask_index():
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    # index by integer
    polygon_masks[0]
    # index by list
    polygon_masks[[0, 1]]
    # index by ndarray
    polygon_masks[np.asarray([0, 1])]
    with pytest.raises(ValueError):
        # invalid index
        polygon_masks[torch.Tensor([1, 2])]


def test_polygon_mask_iter():
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    for i, polygon_mask in enumerate(polygon_masks):
        assert np.equal(polygon_mask, raw_masks[i]).all()
