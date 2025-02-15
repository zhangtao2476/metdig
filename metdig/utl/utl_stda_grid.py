# -*- coding: utf-8 -*-

import datetime

import xarray as xr
import numpy as np
import pandas as pd
from copy import deepcopy
from metpy.units import units

import metdig.utl as mdgstda

__all__ = [
    'xrda_to_gridstda',
    'numpy_to_gridstda',  # 目前不建议用该接口（仍在优化中），改用用npda_to_gridstda
    'npda_to_gridstda',
    'gridstda_full_like',
    'gridstda_full_like_by_levels',
]


def xrda_to_gridstda(xrda,
                     member_dim='member', level_dim='level', time_dim='time', dtime_dim='dtime', lat_dim='lat', lon_dim='lon',
                     member=None, level=None, time=None, dtime=None, lat=None, lon=None,
                     np_input_units='', var_name='',
                     **attrs_kwargs):
    """[将一个xarray数据，转换成网格stda标准格式数据

    通过给出('member', 'level', 'time', 'dtime', 'lat', 'lon')在原始xrda中的维度名称，将xrda转成stda（如果不给出缺失的维度数据，默认填0）

    Example:
    xrda = xr.DataArray([[271, 272, 273], [274, 275, 276]], dims=("X", "Y"), coords={"X": [10, 20], 'Y': [80, 90, 100]})

    # 指定xrda中各个维度对应的stda的维度名称
    stda = metdig.utl.xrda_to_gridstda(xrda, lon_dim='X', lat_dim='Y') 

    # 可以指定缺失的stda维度
    stda = metdig.utl.xrda_to_gridstda(xrda, lon_dim='X', lat_dim='Y', member=['cassandra']) 

    # 可以指定stda的要素，同时给定输入单位，自动转换为stda的单位
    stda = metdig.utl.xrda_to_gridstda(xrda, lon_dim='X', lat_dim='Y', member=['cassandra'], np_input_units='K' ,var_name='tmp') 

    ]

    Args:
        xrda ([xarray.DataArray]): [输入的DataArray]
        member_dim (str, optional): [xrda中代表stda的member维的名称]. Defaults to 'member'.
        level_dim (str, optional): [xrda中代表stda的level维的名称]. Defaults to 'level'.
        time_dim (str, optional): [xrda中代表stda的time维的名称]. Defaults to 'time'.
        dtime_dim (str, optional): [xrda中代表stda的dtime维的名称]. Defaults to 'dtime'.
        lat_dim (str, optional): [xrda中代表stda的lat维的名称]. Defaults to 'lat'.
        lon_dim (str, optional): [xrda中代表stda的lon维的名称]. Defaults to 'lon'.
        member ([list], optional): [使用member该参数替换xrda的member数据]]. Defaults to None.
        level ([list], optional): [使用该参数替换xrda的level数据]. Defaults to None.
        time ([list], optional): [使用该参数替换xrda的time数据]. Defaults to None.
        dtime ([list], optional): [使用该参数e替换xrda的dtime数据]. Defaults to None.
        lat ([list], optional): [使用l该参数替换xrda的lat数据]. Defaults to None.
        lon ([list], optional): [使用该参数替换xrda的lon数据]. Defaults to None.
        np_input_units (str, optional): [输入数据对应的单位，自动转换为能查询到的stda单位]. Defaults to ''.
        var_name (str, optional): [要素名]. Defaults to ''.
        **attrs_kwargs {[type]} -- [其它相关属性，如：data_source='cassandra', level_type='high']

    Returns:
        [STDA] -- [STDA网格数据]
    """
    def _easy_check_None(parm):
        if parm is None:
            return True
        if len(parm) == 1 and parm[0] is None:
            return True
        return False

    stda_data = xrda.copy(deep=True)

    # 已知维度替换成stda维度名称，同时补齐缺失维度
    if member_dim in xrda.dims:
        stda_data = stda_data.rename({member_dim: 'member'})
    else:
        stda_data = stda_data.expand_dims(member=[0])
    if level_dim in xrda.dims:
        stda_data = stda_data.rename({level_dim: 'level'})
    else:
        stda_data = stda_data.expand_dims(level=[0])
    if time_dim in xrda.dims:
        stda_data = stda_data.rename({time_dim: 'time'})
    else:
        stda_data = stda_data.expand_dims(time=[0])
    if dtime_dim in xrda.dims:
        stda_data = stda_data.rename({dtime_dim: 'dtime'})
    else:
        stda_data = stda_data.expand_dims(dtime=[0])
    if lat_dim in xrda.dims:
        stda_data = stda_data.rename({lat_dim: 'lat'})
    else:
        stda_data = stda_data.expand_dims(lat=[0])
    if lon_dim in xrda.dims:
        stda_data = stda_data.rename({lon_dim: 'lon'})
    else:
        stda_data = stda_data.expand_dims(lon=[0])

    # 替换掉需要替换的维度数据
    if _easy_check_None(member) == False:
        stda_data = stda_data.assign_coords(member=member)
    if _easy_check_None(level) == False:
        stda_data = stda_data.assign_coords(level=level)
    if _easy_check_None(time) == False:
        stda_data = stda_data.assign_coords(time=time)
    if _easy_check_None(dtime) == False:
        stda_data = stda_data.assign_coords(dtime=dtime)
    if _easy_check_None(lat) == False:
        stda_data = stda_data.assign_coords(lat=lat)
    if _easy_check_None(lon) == False:
        stda_data = stda_data.assign_coords(lon=lon)

    # 转置到stda维度
    stda_data = stda_data.transpose('member', 'level', 'time', 'dtime', 'lat', 'lon')

    # delete 冗余维度
    stda_data = stda_data.drop([i for i in stda_data.coords if i not in stda_data.dims])

    # attrs
    stda_attrs = mdgstda.get_stda_attrs(var_name=var_name, **attrs_kwargs)
    # 单位转换
    stda_data.values, data_units = mdgstda.numpy_units_to_stda(stda_data.values, np_input_units, stda_attrs['var_units'])
    stda_attrs['var_units'] = data_units
    stda_data.attrs = stda_attrs
    return stda_data


def npda_to_gridstda(npda,
                     dims=('lat', 'lon'),
                     member=None, level=None, time=None, dtime=None, lat=None, lon=None,
                     np_input_units='', var_name='',
                     **attrs_kwargs):
    """[将一个numpy数据，转换成网格stda标准格式数据

    通过给出npda的维度信息及其维度数据，('member', 'level', 'time', 'dtime', 'lat', 'lon')，将npda转成stda（如果不给出缺失的维度数据，默认填0）

    Example:
    npda = np.array([[271, 272, 273], [274, 275, 276]])

    # 指定xrda中各个维度对应的stda的维度名称
    stda = metdig.utl.npda_to_gridstda(npda, dims=('lat', 'lon'), lon=[80, 90, 100], lat=[10, 20])

    # 可以指定缺失的stda维度
    stda = metdig.utl.npda_to_gridstda(npda, dims=('lat', 'lon'), lon=[80, 90, 100], lat=[10, 20], member=['cassandra'])

    # 可以指定stda的要素，同时给定输入单位，自动转换为stda的单位
    stda =  metdig.utl.npda_to_gridstda(npda, dims=('lat', 'lon'), lon=[80, 90, 100], lat=[10, 20], member=['cassandra'], np_input_units='K' ,var_name='tmp') 

    ]

    Args:
        npda ([ndarray]): [numpy数据]
        dims (tuple, optional): [npda对应的stda的维度]. Defaults to ('lat', 'lon').
        member ([list], optional): [npda的member维数据]]. Defaults to None.
        level ([list], optional): [npda的level维数据]. Defaults to None.
        time ([list], optional): [npda的time维数据]. Defaults to None.
        dtime ([list], optional): [npda的dtime维数据]. Defaults to None.
        lat ([list], optional): [npda的lat维数据]. Defaults to None.
        lon ([list], optional): [npda的lon维数据]. Defaults to None.
        np_input_units (str, optional): [np_input数据对应的单位，自动转换为能查询到的stda单位]. Defaults to ''.
        var_name (str, optional): [要素名]. Defaults to ''.
        **attrs_kwargs {[type]} -- [其它相关属性，如：data_source='cassandra', level_type='high']

    Returns:
        [STDA] -- [STDA网格数据]
    """
    if len(npda.shape) != len(dims):
        raise Exception('error: npda shape not equal dims, please check npda and dims')
    for _d in dims:
        if _d != 'member' and _d != 'level' and _d != 'time' and _d != 'dtime' and _d != 'lat' and _d != 'lon':
            raise Exception('''error: dims need the following definitions: ('member', 'level', 'time', 'dtime', 'lat', 'lon'), please check dims''')

    npda = npda.copy()

    temp = dict(member=member, level=level, time=time, dtime=dtime, lat=lat, lon=lon)

    # 第一步：输入的npda转成xrda
    coords = [(_d,  np.arange(_l)) if temp[_d] is None else (_d,  temp[_d]) for _d, _l in zip(dims, npda.shape)]
    xrda = xr.DataArray(npda, coords=coords)

    # 第二步：缺失维度补齐，未指定的维度以0补齐
    for _d in list(set(('member', 'level', 'time', 'dtime', 'lat', 'lon')).difference(set(dims))):
        if temp[_d] is not None:
            xrda = xrda.expand_dims({_d: temp[_d]})
        else:
            xrda = xrda.expand_dims({_d: [0]})

    # 第三步：转置到stda维度
    xrda = xrda.transpose('member', 'level', 'time', 'dtime', 'lat', 'lon')

    # attrs
    stda_attrs = mdgstda.get_stda_attrs(var_name=var_name, **attrs_kwargs)
    # 单位转换
    xrda.values, data_units = mdgstda.numpy_units_to_stda(xrda.values, np_input_units, stda_attrs['var_units'])
    stda_attrs['var_units'] = data_units
    xrda.attrs = stda_attrs

    return xrda


def numpy_to_gridstda(np_input, members, levels, times, dtimes, lats, lons,
                      np_input_units='', var_name='',
                      **attrs_kwargs):
    '''

    [numpy数组转stda网格标准格式]

    Arguments:
        np_input {[ndarray]} -- [numpy数据,维度必须为('member', 'level', 'time', 'dtime', 'lat', 'lon')]
        members {[list or ndarray]} -- [成员列表]
        levels {[list or ndarray]} -- [层次列表]
        times {[list] or ndarray} -- [起报时间列表]
        dtimes {[list or ndarray]} -- [预报失效列表]
        lats {[list or ndarray]} -- [纬度列表]
        lons {[list or ndarray]} -- [经度列表]
        **attrs_kwargs {[type]} -- [其它相关属性，如：data_source='cassandra', level_type='high']

    Keyword Arguments:
        np_input_units {[str]} -- [np_input数据对应的单位，自动转换为能查询到的stda单位]
        var_name {str} -- [要素名] (default: {''})

    Returns:
        [STDA] -- [STDA网格数据]
    '''

    np_input = np_input.copy()

    # get attrs
    stda_attrs = mdgstda.get_stda_attrs(var_name=var_name, **attrs_kwargs)

    # 单位转换
    data, data_units = mdgstda.numpy_units_to_stda(np_input, np_input_units, stda_attrs['var_units'])

    stda_attrs['var_units'] = data_units

    members = np.array(members)
    levels = np.array(levels)
    times = np.array(times)
    dtimes = np.array(dtimes)
    lats = np.array(lats)
    lons = np.array(lons)

    '''
    弃用Dataset
    # create STDA xarray.Dataset
    stda_data = xr.Dataset()
    stda_data['data'] = (['member', 'level', 'time', 'dtime', 'lat', 'lon'], data, stda_attrs)

    stda_data.coords['member'] = ('member', members)
    stda_data.coords['level'] = ('level', levels)
    stda_data.coords['time'] = ('time', times)
    stda_data.coords['dtime'] = ('dtime', dtimes)
    stda_data.coords['lat'] = ('lat', lats)
    stda_data.coords['lon'] = ('lon', lons)
    '''

    # create STDA xarray.DataArray
    coords = [('member', members),
              ('level', levels),
              ('time', times),
              ('dtime', dtimes),
              ('lat', lats),
              ('lon', lons), ]
    stda_data = xr.DataArray(data, coords=coords)
    stda_data.attrs = stda_attrs

    return stda_data


def gridstda_full_like(a, fill_value, dtype=None, var_name='', **attrs_kwargs):
    '''

    [返回一个和参数a具有相同维度信息的STDA数据，并且均按fill_value填充该stda]

    Arguments:
        a {[stda]} -- [description]
        fill_value {[scalar]} -- [Value to fill the new object with before returning it]
        **attrs_kwargs {[type]} -- [其它相关属性，如：data_source='cassandra', level_type='high', data_name='ecmwf']

    Keyword Arguments:
        dtype {[dtype, optional]} -- [dtype of the new array. If omitted, it defaults to other.dtype] (default: {None})
        var_name {str} -- [要素名] (default: {''})

    Returns:
        [stda] -- [stda网格数据]
    '''
    stda_data = xr.full_like(a, fill_value, dtype=dtype)
    stda_data.attrs = mdgstda.get_stda_attrs(var_name=var_name, **attrs_kwargs)
    return stda_data


def gridstda_full_like_by_levels(a, levels, dtype=None, var_name='pres', **attrs_kwargs):
    '''

    [返回一个和参数a具有相同维度信息的stda数据，并且按参数levels逐层赋值]

    Arguments:
        a {[type]} -- [description]
        levels {[type]} -- [description]
        **attrs_kwargs {[type]} -- [其它相关属性，如：data_source='cassandra', level_type='high', data_name='ecmwf']

    Keyword Arguments:
        dtype {[dtype, optional]} -- [dtype of the new array. If omitted, it defaults to other.dtype] (default: {None})
        var_name {str} -- [要素名] (default: {'pres'})

    Returns:
        [stda] -- [stda网格数据]
    '''

    # 后续可以改为stda_broadcast_levels， xr.broadcast(a, levels.squeeze())
    stda_data = gridstda_full_like(a, 0, var_name=var_name, **attrs_kwargs)
    for i, lev in enumerate(levels):
        stda_data.values[:, i, :, :, :, :] = lev
    return stda_data


@xr.register_dataarray_accessor('stda')
class __STDADataArrayAccessor(object):
    """
    stda 格式说明: 维度定义为(member, level, time, dtime, lat, lon)
    """

    def __init__(self, xr):
        self._xr = xr

    @property
    def level(self):
        """[get level]

        Returns:
            [pd.series]: [level]
        """
        return pd.Series(self._xr['level'].values)

    @property
    def fcst_time(self):
        """[get fcst_time*dtime)]

        Returns:
            [pd.series]: [fcst_time]
        """
        fcst_time = []
        for time in self._xr['time'].values:
            for dtime in self._xr['dtime'].values:
                _ = pd.to_datetime(time).replace(tzinfo=None).to_pydatetime() + datetime.timedelta(hours=int(dtime))
                fcst_time.append(_)
        return pd.Series(fcst_time)

    @property
    def time(self):
        """[get time]

        Returns:
            [pd.series]: [time]
        """
        time = pd.to_datetime(self._xr['time'].values)
        return pd.Series(time)

    @property
    def dtime(self):
        """[get dtime]

        Returns:
            [pd.series]: [dtime]
        """
        return pd.Series(self._xr['dtime'].values)

    @property
    def lat(self):
        """[get lat]

        Returns:
            [pd.series]: [lat]
        """
        return pd.Series(self._xr['lat'].values)

    @property
    def lon(self):
        """[get lon]

        Returns:
            [pd.series]: [lon]
        """
        return pd.Series(self._xr['lon'].values)

    @property
    def member(self):
        """[get member]

        Returns:
            [pd.series]: [member]
        """
        return pd.Series(self._xr['member'].values)

    @property
    def values(self):
        """[get values]

        Returns:
            [numpy]: [values]
        """
        return self._xr.values

    @property
    def quantity(self):
        """[get quantity values]

        Returns:
            [quantity numpy]: [quantity values]
        """
        return self.values * units(self._xr.attrs['var_units'])

    @values.setter
    def values(self, values):
        """[set values（注意，该方法为直接赋值不会改变属性信息，如果需要改变属性属性，请调用set_values方法）
        example: sample.stda.values = 1 ]

        Args:
            values ([int or float or numpy]]): [values]
        """
        self._xr.values = values

    def set_values(self, values, var_name=None, **attrs_kwargv):
        """[set values，如果给定var_name。则自动赋值stda属性]

        Args:
            values ([int or float or numpy]): [values]
            var_name ([str], optional): [stda要素名]. Defaults to None.
            **attrs_kwargs {[type]} -- [其它相关属性，如：data_source='cassandra', level_type='high']
        """
        self._xr.values = values
        if var_name is not None:
            attrs = mdgstda.get_stda_attrs(var_name=var_name, **attrs_kwargv)
            self._xr.attrs = attrs

    def get_dim_value(self, dim_name):
        """[获取维度数据，如果dim_name=='fcst_time'情况下，特殊处理，返回time*dtime]

        Args:
            dim_name ([str]): [维度名]

        Returns:
            [numpy]: [维度值]
        """
        if dim_name == 'fcst_time':
            return self.fcst_time.values
        if dim_name == 'time':
            return self.time.values
        return self._xr[dim_name].values

    def get_value(self, ydim='lat', xdim='lon',grid=False):        
        """[根据维度名获取stda数据，
        注： 
        1、网格stda仅支持二维，非二维stda调用该函数会报错
        2、站点stda为pd.DataFrame，无意义，故忽略xdim ydim两个参数]
        3、grid对于站点类型的stda有效,格点类型忽略
        Returns:
            [numpy]: [values]
        """
        if xdim == 'fcst_time':
            if self._xr['time'].values.size == 1:  # 因为是二维，假如time维长度为1，则有维度的肯定在dtime维
                xdim = 'dtime'
            else:
                xdim = 'time'
        if ydim == 'fcst_time':
            if self._xr['time'].values.size == 1:
                ydim = 'dtime'
            else:
                ydim = 'time'
        xdim2 = self._xr.coords[xdim].dims[0]  # 一个dim可能对应多个coord,所以要取到对应的dim
        ydim2 = self._xr.coords[ydim].dims[0]  # 一个dim可能对应多个coord,所以要取到对应的dim
        data = self._xr.squeeze().transpose(ydim2, xdim2).values
        return data

    def description(self):
        '''
        获取描述信息，格式如下:
        起报时间: Y年m月d日H时
        预报时间: Y年m月d日H时
        预报时效: 小时
        '''
        init_time = self.time[0]
        fhour = self.dtime[0]
        fcst_time = self.fcst_time[0]

        if fhour != 0:
            description = '起报时间: {0:%Y}年{0:%m}月{0:%d}日{0:%H}时\n预报时间: {1:%Y}年{1:%m}月{1:%d}日{1:%H}时\n预报时效: {2}小时'.format(
                init_time, fcst_time, fhour)
        else:
            description = '分析时间: {0:%Y}年{0:%m}月{0:%d}日{0:%H}时\n实况/分析'.format(init_time)
        return description

    def description_point(self, describe=''):
        '''
        获取描述信息，格式如下:
        起报时间: Y年m月d日H时
        [data_name]N小时预报[describe]
        预报点: lon, lat

        起报时间: Y年m月d日H时
        [data_name]实况info
        分析点: lon, lat
        '''
        init_time = self.time[0]
        fhour = self.dtime[0]
        point_lon = self.lon[0]
        point_lat = self.lat[0]
        data_name = self.member[0].upper()

        if(fhour != 0):
            description = '起报时间: {0:%Y}年{0:%m}月{0:%d}日{0:%H}时\n[{1}]{2}小时预报{5}\n预报点: {3}, {4}'.format(
                init_time, data_name, fhour, point_lon, point_lat, describe)
        else:
            description = '分析时间: {0:%Y}年{0:%m}月{0:%d}日{0:%H}时\n[{1}]实况/分析{4}\n分析点: {2}, {3}'.format(
                init_time, data_name, point_lon, point_lat, describe)
        return description

    def where(self, conditon, other=np.nan):
        '''
        根据conditon条件过滤数据，类似于xr.DataArray.where
        '''
        return self._xr.where(conditon, other=other)

    def min(self, dim=None, skipna=True, return_number=True):
        """[Return data by applying min along some dimension(s)]

        Args:
            dim ([str], optional): [Dimension(s) over which to apply min]. Defaults to None.
            skipna ([str], optional): [skip missing values (as marked by NaN)]. Defaults to True.
            return_number ([bool], optional): [是否返回数值，默认仅返回numpy数值，若想返回DataArray请设置为False]. Defaults to True.
            **kwargs {[dict]} -- [Additional keyword arguments passed on to the appropriate array function for calculating min on this object’s data]
        """
        ret = self._xr.min(dim=dim, skipna=skipna)
        if return_number:
            return ret.values.squeeze()
        return ret.squeeze()

    def max(self, dim=None, skipna=True, return_number=True):
        """[Return data by applying max along some dimension(s)]

        Args:
            dim ([str], optional): [Dimension(s) over which to apply max]. Defaults to None.
            skipna ([str], optional): [skip missing values (as marked by NaN)]. Defaults to True.
            return_number ([bool], optional): [是否返回数值，默认仅返回numpy数值，若想返回DataArray请设置为False]. Defaults to True.
        """
        ret = self._xr.max(dim=dim, skipna=skipna)
        if return_number:
            return ret.values.squeeze()
        return ret.squeeze()

    def mean(self, dim=None, skipna=True, return_number=True):
        """[Return data by applying mean along some dimension(s)]

        Args:
            dim ([str], optional): [Dimension(s) over which to apply mean]. Defaults to None.
            skipna ([str], optional): [skip missing values (as marked by NaN)]. Defaults to True.
            return_number ([bool], optional): [是否返回数值，默认仅返回numpy数值，若想返回DataArray请设置为False]. Defaults to True.
        """
        ret = self._xr.mean(dim=dim, skipna=skipna)
        if return_number:
            return ret.values.squeeze()
        return ret.squeeze()

    def interp_tosta(self, lon, lat, id=None, other={}, method='linear'):
        """[插值到站点上，返回stda站点数据]

        Args:
            lon ([number or list], optional): [站点经度]
            lat ([number or list], optional): [站点纬度]
            id ([number or str or list], optional): [站号，不填站号则默认从1开始递增]
            other ([dict], optional): [其它坐标信息，以字典方式传参，值可以是列表可以是值，如：other={'city': '北京', 'province': '北京'}]. Defaults to {}.
            method ([str], optional): [interp function(linear or nearest) ]. Defaults to linear.
        """
        def _to_list(parm):
            if isinstance(parm, list):
                return parm
            if isinstance(parm, str):
                return [parm]  # 字符串不能用list()，直接转成list
            try:
                return list(parm)  # numpy or pandas.series
            except:
                return [parm]  # 单项转

        lon = np.array(_to_list(lon))
        lat = np.array(_to_list(lat))

        if id is None:
            id = np.arange(1, lon.size + 1)

        # 其它坐标信息名称
        points_keys = list(set(other.keys()).difference(set(['lon', 'lat', 'id'])))
        points = {k: _to_list(other[k]) for k in points_keys}

        # get points data
        points_xr = self._xr.interp(lon=('points', lon), lat=('points', lat), method=method)
        # print(points_xr)
        # print(points_xr.values.shape)

        # get attrs
        attrs = deepcopy(self._xr.attrs)
        attrs['data_start_columns'] = 6 + len(points_keys)

        # points data to pd.DataFrame
        columns = ['level', 'time', 'dtime', 'id', 'lon', 'lat'] + points_keys + list(self._xr['member'].values)
        lines = []
        for i_lv, _lv in enumerate(points_xr['level'].values):
            for i_t, _t in enumerate(points_xr['time'].values):
                for i_d, _d in enumerate(points_xr['dtime'].values):
                    _d = int(_d)
                    for i_id, _id in enumerate(id):
                        _other = [points[_o][i_id] for _o in points_keys]  # 除去lon lat id之外的其它坐标信息名称对应的数据
                        _lon = lon[i_id]
                        _lat = lat[i_id]
                        _data = points_xr.values[:, i_lv, i_t, i_d, i_id]
                        line = [_lv, _t, _d, _id, _lon, _lat] + _other + list(_data)
                        lines.append(line)

        df = pd.DataFrame(lines, columns=columns)
        df.attrs = attrs

        return df


if __name__ == '__main__':
    pass
