# -*- coding:utf-8 -*-
'''
常用工具.

Version 1.0  2019-09-19 15:56:15 by QiJi
'''
import os
import time
import numpy as np
from tqdm import tqdm
from osgeo import gdal, gdalnumeric  # , osr
import PIL


# **********************************************
# ********** Global default settings ***********
# **********************************************
# CREAT_OPTS = ['COMPRESS=DEFLATE', 'PREDICTOR=2']   # Better zip ratio
CREAT_OPTS = ['COMPRESS=DEFLATE', 'PREDICTOR=1', 'ZLEVEL=1', 'NUM_THREADS=2']  # Speed and zip ratio trade-off
OPEN_OPTS = ['NUM_THREADS=8']


# **********************************************
# *************** Basic tools ******************
# **********************************************
def time_str():
    return time.strftime("%H:%M:%S", time.localtime())


def mkdir_nonexist(path):
    ''' Create a folder if it does not exist. That path can be return. '''
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def filelist(floder_dir, ifPath=False, extension=None):
    '''
    Get names(or whole path) of all files(with specify extension)
    in the floder_dir and return as a list.

    Args:
        floder_dir: The dir of the floder_dir.
        ifPath:
            True - Return whole path of files.
            False - Only return name of files.(Defualt)
        extension: Specify extensions (list) to only get that kind of file names.

    Returns:
        namelist: Name(or path) list of all files(with specify extension)
    '''
    if type(extension) != list:
        extension = [extension]

    namelist = sorted(os.listdir(floder_dir))

    if ifPath:
        for i in range(len(namelist)):
            namelist[i] = os.path.join(floder_dir, namelist[i])

    if extension is not None:
        n = len(namelist)-1  # orignal len of namelist
        for i in range(len(namelist)):
            # if not namelist[n-i].endswith(extension):
            if namelist[n-i].split('.')[-1] not in extension:
                namelist.remove(namelist[n-i])  # discard the files with other extension

    return namelist


def modify_label(label, class_set, fixed_value=None):
    """ 将`label`中不正常（没有被`class_set`所包含的）的值设置为`fixed_value`.
    label is an HW ndarray.
    Args:
        class_set: a dict of categories or list of class num.
    """
    if type(class_set) == dict:
        class_set = [key for (_, key) in class_set.items()]
    if type(class_set) != list:
        ValueError('class_set should be a dict of categories or list of class num')
    if fixed_value is None:
        fixed_value = class_set[-1]

    if 255 not in class_set:
        label[label == 255] = fixed_value
    for ii in range(0, np.max(label)+1):
        # if np.sum(label == ii):
        #     print('err value: %d' % ii)
        if ii not in class_set:
            label[label == ii] = fixed_value
    return label


def changelabel(label, mapping):
    ''' Map the categories using input mapping,
    where mapping is dict{orignal_category1: target_category1, ...}
    Note: Input label is [HW]
    '''
    # label_org = label
    label_new = label.copy()

    for (k, v) in mapping.items():
        label_new[label == k] = v
    return label_new


# **********************************************
# *************** GDAL tools ******************
# **********************************************
def get_gdal_driver(fileformat):
    fileformat = fileformat.split('.')[-1]
    if fileformat.lower() in ['tif', 'tiff']:
        gDriver = gdal.GetDriverByName('GTiff')
    elif fileformat.lower() in ['png']:
        gDriver = gdal.GetDriverByName('PNG')
    elif fileformat.lower() in ['jpg', 'jpeg']:
        gDriver = gdal.GetDriverByName('JPEG')
    return gDriver


def imageToArray(image):
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    array = gdalnumeric.fromstring(image.tobytes(), 'b')
    array.shape = image.im.size[1], image.im.size[0]
    return array


def arrayToImage(array):
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    im = PIL.Image.frombytes('L', (array.shape[1], array.shape[0]),
                             (array.astype('b')).tobytes())
    return im


def array2raster(img_path, array, geotransform=None, geoprojection=None, create_options=[]):
    """ Create raster (file) from array.
    TODO: 还不支持无地理信息模式下写入
    """
    # Get array basic info
    if 'uint8' in array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in array.dtype.name:
        datatype = gdal.GDT_Int16
    elif 'uint16' in array.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'int32' in array.dtype.name:
        datatype = gdal.GDT_Int32
    elif 'uint32' in array.dtype.name:
        datatype = gdal.GDT_UInt32
    elif 'float32' in array.dtype.name:
        datatype = gdal.GDT_Float32
    elif 'float64' in array.dtype.name:
        datatype = gdal.GDT_Float64

    if len(array.shape) == 3:
        rows, cols, band_num = array.shape
    else:
        band_num, (rows, cols) = 1, array.shape

    # Save
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(img_path, cols, rows, band_num, datatype, options=create_options)
    # outRaster = driver.Create(img_path, cols, rows, band_num, options=create_options)

    if geotransform is not None:
        outRaster.SetGeoTransform(geotransform)
    if geoprojection is not None:
        outRaster.SetProjection(geoprojection)

    if len(array.shape) == 2:
        outRaster.GetRasterBand(1).WriteArray(array)
    else:
        for i in range(array.shape[2]):
            outRaster.GetRasterBand(i+1).WriteArray(array[:, :, i])

    outRaster = None


def raster2array(img_path, geoInfo=False, bands=None):
    """ Load raster image from `img_path` and transform to ndarray.
    Returns:
        if geoInfoarray is Ture: array, geotransform, geoprojection
        if geoInfoarray is Flase: array
    Note: 会将整个数据影像全部读入内存，需要注意图像大小与可用RAM
    """
    # raster = gdal.Open(img_path, options=OPEN_OPTS)
    raster = gdal.Open(img_path)
    # Get Geo-Information
    band_num = raster.RasterCount
    geotransform = list(raster.GetGeoTransform())
    geoprojection = raster.GetProjection()

    # Get array
    if band_num == 1:
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
    elif bands is None:
        array = raster.ReadAsArray()
        array = array.transpose((1, 2, 0))
    else:
        arrays = []
        for i in range(raster.RasterCount):
            if i+1 in bands:
                band = raster.GetRasterBand(i+1)
                arrays.append(band.ReadAsArray())
        array = np.stack(arrays, axis=2)

    raster = None
    if geoInfo:
        return array, geotransform, geoprojection
    else:
        return array


def pixel2world(geoMatrix, col_row_st):
    """ Calculate the new geospatial coordinate by using
    gdal geomatrix (gdal.GetGeoTransform()) and pixel location in image.
    Returns:
        a tuple of geoTransform
    """
    col_start, row_start = col_row_st
    if type(geoMatrix) is tuple:
        geoMatrix = list(geoMatrix)  # tuple can't be copied
    new_geoMatrix = geoMatrix.copy()
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    # rtnX = geoMatrix[2]
    # rtnY = geoMatrix[4]
    # pixel = int((x - ulX) / xDist)
    # line = int((ulY - y) / yDist)
    new_geoMatrix[0] = ulX + col_start * xDist
    new_geoMatrix[3] = ulY + row_start * yDist
    return new_geoMatrix


def world2pixel(geoMatrix, xy):
    """ Calculate the pixel location in image by
    gdal geomatrix (gdal.GetGeoTransform()).

    Returns:
        (pixel, line) - (W, H)
    """
    x, y = xy
    ulx = geoMatrix[0]
    uly = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    # rtnX = geoMatrix[2]
    # rtnY = geoMatrix[4]
    pixel = round((x - ulx) / xDist)
    line = round((uly - y) / abs(yDist))

    return (pixel, line)


# **********************************************
# ************ Image basic tools ***************
# **********************************************
def pad_img(image, kh, kw, sh, sw):
    '''Pad image according kernel size and stride.
    Args:
        image - array
        kh, kw - kernel height, kernel width
        sh, sw - height directional stride, width directional stride.
    '''
    h, w = image.shape[:2]
    d = len(image.shape)
    pad_h, pad_w = sh - (h - kh) % sh, sw - (w - kw) % sw
    pad_h = (pad_h // 2, pad_h // 2 + 1) if pad_h % 2 else (pad_h // 2,
                                                            pad_h // 2)
    pad_w = (pad_w // 2, pad_w // 2 + 1) if pad_w % 2 else (pad_w // 2,
                                                            pad_w // 2)
    pad_params = (pad_h, pad_w) if d == 2 else (pad_h, pad_w, (0, 0))
    return np.pad(image, pad_params, mode='constant'), pad_params


def resampled(image_dir, out_dir=None, scale=1 / 5, create_options=None):
    """
    Resample all the images under `image_dir` for `scale` and output the result at `out_dir`.
    Default is Nearest.

    Args:
        scale: newH = orgH * scale, newW = orgW * scale
    """
    if out_dir is None:
        out_dir = image_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if create_options is None:
        create_options = CREAT_OPTS

    image_names = sorted(os.listdir(image_dir))
    for name in tqdm(image_names):
        src_ds = gdal.Open(image_dir + '/' + name, -1)
        out_rows = int(src_ds.RasterYSize * scale)  # scale=1/5: 2m -> 10m
        out_cols = int(src_ds.RasterXSize * scale)
        num_bands = src_ds.RasterCount
        dataType = src_ds.GetRasterBand(1).DataType

        gtiff_driver = gdal.GetDriverByName('GTiff')
        out_ds = gtiff_driver.Create(out_dir + '/' + name, out_cols, out_rows,
                                     num_bands, dataType, options=create_options)
        out_ds.SetProjection(src_ds.GetProjection())
        geotransform = list(src_ds.GetGeoTransform())
        geotransform[1] /= scale
        geotransform[5] /= scale
        out_ds.SetGeoTransform(geotransform)

        dst_ds = src_ds.ReadRaster(buf_xsize=out_cols, buf_ysize=out_rows)
        out_ds.WriteRaster(0, 0, out_cols, out_rows, dst_ds)
        out_ds.FlushCache()


def colour_code_label(label, label_values):
    '''
    Given a [HW] array of class keys, colour code the label;
    also can weight the colour coded label and image, maybe save the final result.

    Args:
        label: single channel array where each value represents the class key.
        label_values: list of [R,G,B] for per category, [[r1,g1,b1], [r2,g2,b2], ...]
    Returns:
        Colour coded label - a ndarray of [HW3].
    '''
    colour_codes = np.array(label_values)
    if len(label) == 3:
        label = label[0]  # [HWC] -> [HW]
    color_label = colour_codes[label.astype(int)]
    color_label = color_label.astype(np.uint8)

    return color_label


def linear_stretch(image, scale=0.01, channel_wise=False):
    # dtype = image.dtype
    if image.shape == 2 or (not channel_wise):
        minL, maxL = 0, 0
        hist = np.unique(image)

        pixel_total = image.size
        num_pixel_low = int(pixel_total * scale)
        num_pixel_up = int(pixel_total * (1-scale))

        minL, maxL = 0, 0
        hist = np.bincount(image.flatten())
        pixel_cumul = 0
        for L, num in enumerate(hist):
            pixel_cumul += num
            if pixel_cumul >= num_pixel_low:
                minL = L
                break

        pixel_cumul = pixel_total
        for L in range(0, hist.shape[0])[::-1]:
            pixel_cumul -= num
            if pixel_cumul <= num_pixel_up:
                maxL = L
                break

        image = np.clip((255.0*(image-minL)/(maxL-minL)), 0, 255).astype(np.uint8)
    else:
        H, W, band_num = image.shape
        new_image = []
        for b in range(band_num):
            band = image[:, :, b]
            pixel_total = band.size
            num_pixel_low = int(pixel_total * scale)
            num_pixel_up = int(pixel_total * (1-scale))

            minL, maxL = 0, 0
            hist = np.bincount(band.flatten())
            pixel_cumul = 0
            for L, num in enumerate(hist):
                pixel_cumul += num
                if pixel_cumul >= num_pixel_low:
                    minL = L
                    break

            pixel_cumul = pixel_total
            for L in range(0, hist.shape[0])[::-1]:
                pixel_cumul -= hist[L]
                if pixel_cumul <= num_pixel_up:
                    maxL = L
                    break
            # band = np.clip(band, minL, maxL)
            band = np.clip(255.0*(band-minL)/(maxL-minL), 0, 255).astype(np.uint8)
            new_image.append(band)
        image = np.stack(new_image, axis=2)
    return image


def histogram(array, L=256):
    """
    Histogram function for multi-dimensional array.
    a = array
    bins = range of numbers to match
    """
    bins = range(0, L)
    fa = array.flat
    n = gdalnumeric.searchsorted(gdalnumeric.sort(fa), bins)
    n = gdalnumeric.concatenate([n, [len(fa)]])
    hist = n[1:]-n[:-1]
    return hist


def hist_stretch(image, L=256):
    """
    Performs a histogram stretch on a gdalnumeric array image.
    image: PIL.Image object
    """
    import operator
    from functools import reduce

    hist = histogram(image)
    im = arrayToImage(image)
    lut = []
    for b in range(0, len(hist), L):
        # step size
        step = reduce(operator.add, hist[b:b+L]) / (L-1)
        # create equalization lookup table
        n = 0
        for i in range(L):
            lut.append(n / step)
            n = n + hist[i+b]
        im = im.point(lut)
    return imageToArray(im)


# **********************************************
# ************* Usefull functions **************
# **********************************************
def crop_by_rowcol(img_path, save_path, crop_info, create_options=None, fileformat='tif'):
    ''' Crop patch(s) from big image with specific row and cols information and save it.
    Note: row - Y axis，col - X axis
    Args:
        save_path: file path(s) to save cropped result, one string of a list of strings for multi-cropping.
        crop_info: one crop_info [row_start, col_start, out_H, out_W] for single cropping,
            or a list of crop_info for multi-cropping.

    '''
    if type(crop_info[0]) != list:
        crop_info = [crop_info]
    if type(save_path) != list:
        save_path = [save_path]
    assert len(crop_info) == len(save_path)
    for spath in save_path:
        mkdir_nonexist(os.path.dirname(spath))

    if create_options is None:
        create_options = CREAT_OPTS

    src_ds = gdal.Open(img_path)
    if src_ds is None:
        ValueError('Cannot open %s!' % img_path)

    # Get basic information about image
    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    num_bands = src_ds.RasterCount

    dataType = src_ds.GetRasterBand(1).DataType
    # print('type', dataType)

    geoProj = src_ds.GetProjection()
    geoTrans = src_ds.GetGeoTransform()

    # Crop
    for (s_pth, c_inf) in zip(save_path, crop_info):
        fname, extn = os.path.splitext(s_pth)

        row_start, col_start, out_H, out_W = c_inf

        if row_start >= rows or row_start < 0:
            ValueError('The specified row_start (%f) is outside the image.' % row_start)
        if col_start >= cols or col_start < 0:
            ValueError('The specified col_start (%f) is outside the image.' % col_start)
        if col_start + out_W >= cols or row_start + out_H >= rows:
            ValueError('The specified crop area (%f, %f) is out of the image range.' % (
                       out_H, out_W))

        patch_ds = src_ds.ReadRaster(xoff=col_start, yoff=row_start, xsize=out_W, ysize=out_H)
        # Get image dirver
        gDriver = get_gdal_driver(extn[1:])

        # Save
        out_ds = gDriver.Create(s_pth, out_W, out_H, num_bands, dataType, options=create_options)
        out_ds.SetProjection(geoProj)
        out_ds.SetGeoTransform(pixel2world(geoTrans, col_start, row_start))

        out_ds.WriteRaster(0, 0, out_W, out_H, patch_ds)
        out_ds.FlushCache()


def crop_by_coords(img_path, save_path, crop_info, create_options=None):
    """ Crop a path from big image with specific Lan and Lng infomation and save it (in same fileformat).
    Multi-cropping is supported.
    Note: lat - X axis - W, lng - Y axis - H

    Args:
        save_path: file path(s) to save cropped result, one string of a list of strings for multi-cropping.
        crop_info: one crop_info - [lat_start, lng_start, out_lat, out_lng]
            or [lat_start, lng_start, out_lat, out_lng, out_W, out_H] for single cropping,
            or a list of crop_info for multi-cropping.
    """
    if type(crop_info[0]) != list:
        crop_info = [crop_info]
    if type(save_path) != list:
        save_path = [save_path]
    assert len(crop_info) == len(save_path)

    stride = 40000  # Set max stride to load into RAM

    for spath in save_path:
        mkdir_nonexist(os.path.dirname(spath))
    if create_options is None:
        create_options = CREAT_OPTS

    # Open file
    src_ds = gdal.Open(img_path)
    if src_ds is None:
        ValueError('Cannot open %s!' % (img_path))

    # Get basic information about image
    cols = src_ds.RasterXSize  # W
    rows = src_ds.RasterYSize  # H
    num_bands = src_ds.RasterCount  # C

    dataType = src_ds.GetRasterBand(1).DataType
    # print('type', dataType)

    geoProj = src_ds.GetProjection()
    geoTrans = src_ds.GetGeoTransform()
    ulX, xDist, rtnX, ulY, rtnY, yDist = geoTrans
    if rtnX != 0 or rtnY != 0:
        ValueError('Roated image is not supported yet!')

    out_crop_info = []

    # Crop
    for (s_pth, c_inf) in zip(save_path, crop_info):
        fname, extn = os.path.splitext(s_pth)
        # Cal image coordinates based on latitude and longitude
        if len(c_inf) == 4:
            lat_start, lng_start, out_lat, out_lng = c_inf
            out_W = round(out_lat / xDist)
            out_H = round(-out_lng / yDist)
        elif len(c_inf) == 6:
            lat_start, lng_start, out_lat, out_lng, out_W, out_H = c_inf
        else:
            ValueError('Wrong length of crop_info!')

        # Cal geoTransforms
        x_start = round((lat_start - ulX) / xDist)  # 无论如何都会产生偏移
        y_start = round((lng_start - ulY) / yDist)  # TODO:考虑保留小数位的像素用于修正裁剪后的影像地理信息

        # TODO: If roate?
        if x_start >= cols:
            ValueError('The specified start_lng (%f) is outside the image.' % lat_start)
        elif x_start < 0:
            out_W += x_start
            x_start, lat_start = 0, ulX
            # crop_by_coords()
        if y_start >= rows:
            ValueError('The specified start_lng (%f) is outside the image.' % lng_start)
        elif y_start < 0:
            out_H += y_start
            y_start, lng_start = 0, ulY
        out_W = min(out_W, cols - x_start)
        out_H = min(out_H, rows - y_start)
        # if x_start + out_W >= cols or y_start + out_H >= rows:
        #     ValueError('The specified crop area (%f, %f) is out of the image range.' % (out_lat, out_lng))

        geoTrans = [lat_start, xDist, rtnX, lng_start, rtnY, yDist]  # update geotrans

        # Get image dirver
        gDriver = get_gdal_driver(extn[1:])

        # Save
        out_ds = gDriver.Create(s_pth, out_W, out_H, num_bands, dataType, options=create_options)
        out_ds.SetProjection(geoProj)
        out_ds.SetGeoTransform(geoTrans)

        if out_W > stride or out_H > stride:
            # 如果裁剪区域过大则需要分块儿读取
            for y in range(0, out_H, stride):
                y_size = min(stride, out_H - y)
                for x in range(0, out_W, stride):
                    x_size = min(stride, out_W - x)
                    patch_ds = src_ds.ReadRaster(xoff=x+x_start, yoff=y+y_start, xsize=x_size, ysize=y_size)
                    out_ds.WriteRaster(x, y, x_size, y_size, patch_ds)
        else:
            patch_ds = src_ds.ReadRaster(xoff=x_start, yoff=y_start, xsize=out_W, ysize=out_H)
            out_ds.WriteRaster(0, 0, out_W, out_H, patch_ds)
        out_ds.FlushCache()

        out_crop_info.append([lat_start, lng_start, 0, 0, out_W, out_H])

    return out_crop_info


def crop_by_masks(target_path, mask_path, save_path, create_options=None):
    """ Crop a patch from big image with specific image with coordinates info and save it.

    Args:
        target_path: The file path of target image to be croped.
        mask_path: The file path(s) of mask image to crop the target,
            could be a list of paths of mask files for multi-cropping.
        save_path: file path(s) to save cropped result.
    """
    out_dir = os.path.dirname(save_path)
    mkdir_nonexist(out_dir)

    # Get crop info. from mask
    if type(mask_path) != list:
        mask_path = [mask_path]
    if type(save_path) != list:
        save_path = [save_path]
    assert len(mask_path) == len(save_path)

    if create_options is None:
        create_options = CREAT_OPTS

    crop_infos = []
    for (m_pth, s_pth) in zip(mask_path, save_path):
        # Open file
        src_ds = gdal.Open(m_pth)
        if src_ds is None:
            ValueError('Cannot open mask: %s!' % (m_pth))

        cols = src_ds.RasterXSize  # W
        rows = src_ds.RasterYSize  # H
        ulX, xDist, rtnX, ulY, rtnY, yDist = src_ds.GetGeoTransform()
        crop_infos.append([ulX, ulY, xDist*cols, -yDist*rows, cols, rows])

    out_crop_info = crop_by_coords(target_path, save_path, crop_infos, create_options)
    # 如果 mask无法被target完全包含，则取两者交集
    new_save_path = [mp.replace('Org_images', 'Org_images_extract') for mp in mask_path]
    # new_save_path = r'G:/Data Bank/GLC/Org_images_test/'
    for m_pth, s_pth, c_info in zip(mask_path, new_save_path, out_crop_info):
        src_ds = gdal.Open(m_pth)
        cols = src_ds.RasterXSize  # W
        rows = src_ds.RasterYSize  # H
        if cols == c_info[4] and rows == c_info[5]:
            continue
        crop_by_coords(m_pth, s_pth, c_info, [])
    # Finish


def crop_imgs(image_dir,
              crop_params,
              out_dir=None,
              create_options=None,
              fileformat='tif',
              log_path=None,
              ifPad=False):
    ''' Slide crop images into small piece and save them.
    Note: not pad yet.
    '''
    if out_dir is None:
        out_dir = image_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    crop_h, crop_w, stride = crop_params
    if create_options is None:
        create_options = CREAT_OPTS
    gDriver = get_gdal_driver(fileformat)
    if '.' not in fileformat:
        fileformat = '.' + fileformat

    name2world = {}  # log dict
    image_names = sorted(os.listdir(image_dir))[:1]
    # for name in tqdm(image_names):
    for name in image_names:
        # Open dataset (not load into RAM)
        # st = time.time()
        src_ds = gdal.Open(image_dir + '/' + name)
        if src_ds is None:
            ValueError('Cannot open %s!' % (image_dir + '/' + name))
        # print('Time cost - Open: %.4fs' % (time.time() - st))

        # Get basic information about image
        fname, _ = os.path.splitext(name)

        cols = src_ds.RasterXSize  # W
        rows = src_ds.RasterYSize  # H
        num_bands = src_ds.RasterCount  # C

        dataType = src_ds.GetRasterBand(1).DataType
        # print('type', dataType)

        geoProj = src_ds.GetProjection()
        geoTrans = src_ds.GetGeoTransform()

        # Crop and save
        y = 0  # y is H(row)
        for i in range((rows-crop_h)//stride + 1):
            x = 0  # x is W(col)
            for j in range((cols - crop_w)//stride + 1):
                # crop and load into RAM (10000x10000x3 uint8 is about 500MB)
                patch_ds = src_ds.ReadRaster(xoff=x,
                                             yoff=y,
                                             xsize=crop_w,
                                             ysize=crop_h)
                # save
                save_name = '%s_%d_%d%s' % (fname, (i+1), (j+1), fileformat)
                out_ds = gDriver.Create(
                    out_dir+'/'+save_name, crop_h, crop_w, num_bands, dataType,
                    options=create_options)

                out_ds.SetProjection(geoProj)
                out_ds.SetGeoTransform(pixel2world(geoTrans, x, y))

                out_ds.WriteRaster(0, 0, crop_h, crop_w, patch_ds)
                out_ds.FlushCache()

                if log_path is not None:
                    name2world[save_name] = [geoTrans, geoProj]
                if not os.path.exists(out_dir+'/'+save_name):
                    print('Fail to create file - %s!' % (out_dir+'/'+save_name))

                x += stride
            y += stride
    if log_path is not None:
        with open(log_path, 'w') as f:
            f.write(name2world)


def labels_visualize(label_dir,
                     mCategory,
                     out_dir=None,
                     create_options=None,
                     fileformat='tif'):
    """ Visualize all labels in `label_dir`.
    Note: mCategory can be seen in category.py
    """
    if out_dir is None:
        out_dir = label_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if create_options is None:
        create_options = CREAT_OPTS

    for name in sorted(os.listdir(label_dir)):
        # Load data
        lbl_ds = gdal.Open(label_dir + '/' + name)
        lbl_band = lbl_ds.GetRasterBand(1)  # 只取一个波段
        lbl = lbl_band.ReadAsArray()

        geoProj = lbl_ds.GetProjection()
        geoTrans = lbl_ds.GetGeoTransform()

        # Visualization
        lbl = modify_label(lbl, mCategory.table, mCategory.num - 1)
        lbl_vis = colour_code_label(lbl, mCategory.color_table)

        # Save
        gDriver = get_gdal_driver(fileformat)
        out_ds = gDriver.Create(out_dir + '/' + name, lbl.shape[1], lbl.shape[0], 3, options=create_options)
        out_ds.SetGeoTransform(geoTrans)
        out_ds.SetProjection(geoProj)

        for i in range(lbl_vis.shape[2]):
            out_ds.GetRasterBand(i + 1).WriteArray(lbl_vis[:, :, i])

        out_ds = None


def statistics_v0(image_dir, label_dir, class_tabel):
    """ 获得各区域影像数据的统计信息(由于不适合数据模式，暂时被淘汰了).
    Note: image_dir, label_dir are dir of all pairs of image and label,
        each pair of image and label should has same sign in file name.
        For example, 'tianmen_a.tiff' and 'tianmen_a_mask.tiff' both have 'tianmen_a' in their names .
    """
    # Image names list
    # img_names = sorted(os.listdir(image_dir))
    lbl_names = sorted(os.listdir(label_dir))

    # Execution operations
    # for (img_name, lbl_name) in tqdm(zip(img_names, lbl_names)):
    for lbl_name in lbl_names:
        print('\n' + '*' * 50 + '\n' + lbl_name.split('_mask')[0])

        # image = utils.raster2array(image_dir+'/'+img_name)
        label = raster2array(label_dir + '/' + lbl_name)
        # TODO： 一次性统计的话内存可能不够
        # assert image.shape[:2] == label.shape[:2]

        total_pixels = label.size
        for cls_name, ind in class_tabel.items():
            ratio = np.sum(label == ind) / total_pixels

            # Two way of print result:
            #   1. print sample ratio of each class vertically with markdown tabel style
            print('| %.2f%% |' % (ratio * 100))
            #   2. print them horizontally.
            # print('%.4fs, ' % ratio, end='')


def statistics(root, category):
    """ 根据GT_masks获得各区域影像数据的统计信息, 输出总体类别占比，以及各各区域内的类别占比.
    """
    import cv2

    class_num = category.num - 1  # 剔除BG
    regions = [name for name in os.listdir(root)]

    print('| Region num |', end='')
    for c in range(1, category.num):
        print(' %s |' % category.names[c], end='')

    total_pcpn = np.zeros((class_num), dtype=np.int64)  # Per class pixel num
    for reg in regions:
        lbl_list = filelist(root + '/' + reg, True, extension=['png', 'tif'])

        reg_pcpn = np.zeros((class_num), dtype=np.int64)  # Per class pixel num
        for lbl_path in lbl_list:
            lbl = cv2.imread(lbl_path, 0)

            for c in range(1, category.num):
                reg_pcpn[c-1] += np.sum(lbl == c)
        total_pcpn += reg_pcpn
        reg_ratios = reg_pcpn / np.sum(reg_pcpn)
        # 输出结果
        print('\n\r| %s |' % reg, end='')
        for c in range(1, category.num):
            print(' %.2f%% |' % (reg_ratios[c-1] * 100), end='')

    total_ratios = total_pcpn / np.sum(total_pcpn)
    print('\n\r| Total |', end='')
    for c in range(1, category.num):
        print(' %.2f%% |' % (total_ratios[c-1] * 100), end='')


def main():
    print('main')

    from category import category_C as mCategory

    ROOT = '/media/tao/Seagate Expansion Drive'
    # ROOT = r'D:\Data\GLC'
    GT_DIR = ROOT + '/' + 'mask_mid_up8'
    statistics(GT_DIR, mCategory)


if __name__ == "__main__":
    print('utils')
    main()
    pass
