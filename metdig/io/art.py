# -*- coding: utf-8 -*-

import datetime
import os
import sys
import math

import cdsapi
import numpy as np
import xarray as xr

import sys

import metdig.utl as mdgstda

from metdig.io.lib import utility as utl
from metdig.io.lib import art_cfg
from metdig.io.lib import config as CONFIG

import logging
_log = logging.getLogger(__name__)


def get_model_grid(init_time=None, fhour=None, var_name=None, level=None, extent=None, x_percent=0, y_percent=0, **kwargs):
    '''

    [获取art再分析单层单时次数据，注意：缓存的目录为世界时]

    Keyword Arguments:
        init_time {[datetime]} -- [再分析时间（北京时）] (default: {None})
        var_name {[str]} -- [数据要素名] (default: {None})
        level {[int32]} -- [层次，不传代表地面层] (default: {None})
        extent {[tuple]} -- [裁剪区域，如(50, 150, 0, 65)] (default: {None})
        x_percent {number} -- [根据裁剪区域经度方向扩充百分比] (default: {0})
        y_percent {number} -- [根据裁剪区域纬度方向扩充百分比] (default: {0})

    Returns:
        [type] -- [description]
    '''

    init_time_utc = init_time - datetime.timedelta(hours=8)  # 世界时
    if extent:
        # 数据预先扩大xy percent
        delt_x = (extent[1] - extent[0]) * x_percent
        delt_y = (extent[3] - extent[2]) * y_percent
        extent = (extent[0] - delt_x, extent[1] + delt_x, extent[2] - delt_y, extent[3] + delt_y)
        extent = (math.floor(extent[0]), math.ceil(extent[1]), math.floor(extent[2]), math.ceil(extent[3]))
        extent = (
            extent[0] if extent[0] >= -180 else -180,
            extent[1] if extent[1] <= 180 else 180,
            extent[2] if extent[2] >= -90 else -90,
            extent[3] if extent[3] <= 90 else 90,
        )
    else:
        extent = [50, 160, 0, 70]  # 数据下载默认范围

    # 从配置中获取相关信息
    #try:
    if level:
        level_type = 'high'
    #        cache_file = CONFIG.get_artcache_file(init_time_utc, var_name, extent, level=level, find_area=True)
    else:
        level_type = 'surface'
    #        cache_file = CONFIG.get_artcache_file(init_time_utc, var_name, extent, level=None, find_area=True)
    art_var = art_cfg().art_variable(var_name=var_name, level_type=level_type)
    art_vname = art_cfg().art_vname(var_name=var_name, level_type=level_type)
    art_leveltype = art_cfg().art_leveltype(var_name=var_name, level_type=level_type)
    art_level = art_cfg().art_level(var_name=var_name, level_type=level_type, level=level)
    art_units = art_cfg().art_units(level_type=level_type, var_name=var_name)
    print(art_var,art_level,art_units,art_vname,art_leveltype,level,fhour)
    #except Exception as e:
    #    raise Exception(str(e))

    #if not os.path.exists(cache_file):
    #    print("zt cache_file:",cache_file," not exists,exit....")
    #    exit()
    # 此处读到的dataset应该只有一个数据集，维度=[time=1,latitude,longitude]，因为下载的时候均是单层次下载
    #print(var_name,level_type)
    #ds = xr.open_dataset("/home/zt/gdas.t00z.master.grb2f06", engine='cfgrib',backend_kwargs={'filter_by_keys':{'shortName':"t",'typeOfLevel':"isobaricInhPa"}})
    #
    
    if fhour:fh="f%02d"%fhour
    else:fh="anal"
    date10=init_time_utc.strftime("%Y%m%d%H")
    fname0="/mnt/d/ART/gdas."+date10[0:10]+"."+fh+".grib2"
    fname0="/home/zt/gdas.t00z.master.grb2f06"
    fname="/g9/cra_xp/ART_GLB/ATM/6HOURLY/"+date10[0:4]+"/"+date10[0:8]+"/gdas."+date10[0:10]+"."+fh+".grib2"
    print(fname0,os.path.exists(fname0))
    print(fname)
    if not os.path.exists(fname0): 
        print("no file and download from pi:")
        os.system("scp cra_op@10.40.143.19:"+fname+" /mnt/d/ART/")
    print(fname0)
    print(art_vname,art_leveltype)
    if level:
        data = xr.open_dataset(fname0,engine='cfgrib',backend_kwargs={'filter_by_keys':{'shortName':art_vname,'typeOfLevel':art_leveltype,'level':level}})
    else:
        data = xr.open_dataset(fname0,engine='cfgrib',backend_kwargs={'filter_by_keys':{'shortName':art_vname,'typeOfLevel':art_leveltype}})
        #print(data)
    data = data.to_array()
    
    data = data.transpose('variable','latitude', 'longitude')
    data = data.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    # 数据裁剪，此处不传xpercent，因为之前已经扩大范围了时候已经扩大范围了
    data = utl.area_cut(data, extent)
    
    # 经纬度从小到大排序好
    data = data.sortby('lat')
    data = data.sortby('lon')
    #('member', 'level', 'time', 'dtime', 'lat', 'lon')
    #print(data)
    #print(art_level,init_time,var_name,art_units,level_type)
    #print(data.coords['variable'][0])
    data= data.sel(variable=data.coords['variable'][0]).drop('variable')
    #print(data)
    stda_data = mdgstda.xrda_to_gridstda(data,
                                         lat_dim='lat', lon_dim='lon',
                                         member=['art'], level=[art_level], time=[init_time],
                                         var_name=var_name, np_input_units=art_units, dtime=[fhour],
                                         data_source='nmic',  level_type=level_type)
    return stda_data


def get_model_grids(init_times=None, fhour=None, var_name=None, level=None, extent=None, x_percent=0, y_percent=0, **kwargs):
    '''

    [读取单层多时次模式网格数据]

    Keyword Arguments:
        init_times {[list or time]} -- [再分析时间列表] (default: {None})
        var_name {[str]} -- [要素名]
        level {[int32]} -- [层次，不传代表地面层] (default: {None})
        extent {[tuple]} -- [裁剪区域，如(50, 150, 0, 65)] (default: {None})
        x_percent {number} -- [根据裁剪区域经度方向扩充百分比] (default: {0})
        y_percent {number} -- [根据裁剪区域纬度方向扩充百分比] (default: {0})

    Returns:
        [stda] -- [stda格式数据]
    '''
    init_times = utl.parm_tolist(init_times)

    stda_data = []
    for init_time in init_times:
        try:
            data = get_model_grid(init_time, fhour, var_name, level, extent=extent, x_percent=x_percent, y_percent=y_percent)
            if data is not None and data.size > 0:
                stda_data.append(data)
        except Exception as e:
            _log.info(str(e))
    if stda_data:
        return xr.concat(stda_data, dim='time')
    return None


def get_model_3D_grid(init_time=None, fhour=None, var_name=None, levels=None, extent=None, x_percent=0, y_percent=0, **kwargs):
    '''

    [读取多层单时次模式网格数据]

    Keyword Arguments:
        init_time {[datetime]} -- [再分析时间]
        var_name {[str]} -- [要素名]
        levels {[list or number]} -- [层次，不传代表地面层] (default: {None})
        extent {[tuple]} -- [裁剪区域，如(50, 150, 0, 65)] (default: {None})
        x_percent {number} -- [根据裁剪区域经度方向扩充百分比] (default: {0})
        y_percent {number} -- [根据裁剪区域纬度方向扩充百分比] (default: {0})

    Returns:
        [stda] -- [stda格式数据]
    '''
    levels = utl.parm_tolist(levels)

    stda_data = []
    for level in levels:
        try:
            data = get_model_grid(init_time, fhour, var_name, level, extent=extent, x_percent=x_percent, y_percent=y_percent)
            if data is not None and data.size > 0:
                stda_data.append(data)
        except Exception as e:
            _log.info(str(e))

    if stda_data:
        return xr.concat(stda_data, dim='level')
    return None


def get_model_3D_grids(init_times=None, fhour=None, var_name=None, levels=None, extent=None, x_percent=0, y_percent=0, **kwargs):
    '''

    [读取多层多时次模式网格数据]

    Keyword Arguments:
        init_times {[list or time]} -- [再分析时间列表] (default: {None})
        var_name {[str]} -- [要素名]
        levels {[list or number]} -- [层次，不传代表地面层] (default: {None})
        extent {[tuple]} -- [裁剪区域，如(50, 150, 0, 65)] (default: {None})
        x_percent {number} -- [根据裁剪区域经度方向扩充百分比] (default: {0})
        y_percent {number} -- [根据裁剪区域纬度方向扩充百分比] (default: {0})

    Returns:
        [stda] -- [stda格式数据]
    '''
    init_times = utl.parm_tolist(init_times)
    levels = utl.parm_tolist(levels)

    stda_data = []
    for init_time in init_times:
        temp_data = []
        for level in levels:
            try:
                data = get_model_grid(init_time, fhour, var_name, level, extent=extent, x_percent=x_percent, y_percent=y_percent)
                if data is not None and data.size > 0:
                    temp_data.append(data)
            except Exception as e:
                _log.info(str(e))
                continue
        if temp_data:
            temp_data = xr.concat(temp_data, dim='level')
            stda_data.append(temp_data)
    if stda_data:
        return xr.concat(stda_data, dim='time')
    return None


def get_model_points(init_time=None, fhour=None, var_name=None, levels=None, points={}, **kwargs):
    '''

    [读取单层/多层，单时效/多时效 模式网格数据，插值到站点上]

    Keyword Arguments:
        init_times {[list or time]} -- [再分析时间] (default: {None})
        var_name {[str]} -- [要素名]
        levels {[list]} -- [层次，不传代表地面层] (default: {None})
        points {[dict]} -- [站点信息，字典中必须包含经纬度{'lon':[], 'lat':[]}]

    Returns:
        [stda] -- [stda格式数据]
    '''
    levels = utl.parm_tolist(levels)

    # get grids data
    stda_data = get_model_3D_grids(init_time, fhour, var_name, levels)

    if stda_data is not None and stda_data.size > 0:
        return mdgstda.gridstda_to_stastda(stda_data, points)
    return None


if __name__ == '__main__':
    init_time = datetime.datetime(2020, 8, 2, 8)
    # data = get_model_grid(init_time, 'hgt', level=500, extent=[70, 140, 10, 60])
    # extent = [50, 160, 0, 70]
    extent = (70, 140, 15, 55)
    print(extent)
    # data = get_model_grid(init_time, 'hgt', level=500, extent=extent) # 587.42365
    data = get_model_grid(init_time, 'spfh', level=500, extent=extent)  # 587.42365
    print(extent)
    print(data)

    # C:\Users\Administrator\.metart\cache\202008030800\hourly\hgt\500\202008030800_56_154_11_59.nc
    # get_model_grid(init_time, 'hgt', level=100)
    # get_model_grid(init_time, 'hgt', level=200)
    # get_model_grid(init_time, 'u', level=100)

    # get_model_grid(init_time, 'u10m')
