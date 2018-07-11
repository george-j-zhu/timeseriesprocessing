#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

"""
DataFrame Class for time series processing.
"""

from datetime import datetime
from . import utilities
import pandas as pd
import numpy as np


class TimeSeriesDataFrame:
    """
    Data Frame for time series data
    """

    def __init__(self, start_datetime, end_datetime, x_columns, y_columns=None):
        """
        constructor

        Args:
            start_datetime:
            end_datetime:
            columns, dictionary
        """
        self.dataframe = None
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.x_columns = x_columns
        self.y_columns = y_columns


    def __is_initialized(self):
        """
        ckeck if constructor function __init__ was called.

        Return:
            True if initialized.

        Raise:
            Exception: exception errors when TimeSeriesDataFrame Object is not initialized.
        """
        if self.start_datetime is None or self.end_datetime is None:
            raise Exception("TimeSeriesDataFrame Object is not initialized!")

        return True

    def append(self, new_dataframe, axis=0):
        """
        append a new pandas dataframe to this TimeSeriesDataFrame object.
        TODO this function will check whether the new dataaframe to be appended is
        has newer datetimes on the first column then the existing dataframe.

        Returns:
            None
        """
        if axis == 1:
            self.__append_columns(new_dataframe)
            return

        if self.dataframe is None:
            self.dataframe = pd.DataFrame(columns=new_dataframe.columns)

        start_pos = -1
        end_pos = -1
        if self.__is_initialized() is True:
            start_datetime_dt = datetime.strptime(self.start_datetime, "%Y/%m/%d %H:%M")
            end_datetime_dt = datetime.strptime(self.end_datetime, "%Y/%m/%d %H:%M")
            for index, row in new_dataframe.iloc[:, 0:1].iterrows():
                for datetime_format in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
                    try:
                        temp_datetime_dt = datetime.strptime(row[0], datetime_format)
                        break
                    except Exception:
                        continue
                if temp_datetime_dt <= end_datetime_dt and temp_datetime_dt >= start_datetime_dt:
                    # found data in the specified period.
                    end_pos = index
                    if start_pos == -1:
                        start_pos = index
            self.dataframe = self.dataframe.append(
                new_dataframe.iloc[start_pos:end_pos, :], ignore_index=True)


    def __append_columns(self, new_dataframe):
        """
        append a new dataframe as new columns.
        self.dataframe and new_dataframe must contain "time" column, and the format must be the same.
        """
        self.dataframe = pd.merge(self.dataframe, new_dataframe)


    def get_x_dataframe_without_time_column(self):
        """
        extract data as a new dataframe according to columns
        """
        if self.x_columns is None:
            raise Exception

        return utilities.impute_missing_data(self.dataframe.loc[:, self.x_columns])

    def get_x_dataframe_with_time_column(self):
        """
        extract data as a new dataframe according to columns
        """
        return pd.concat([self.get_time_dataframe(), self.get_x_dataframe_without_time_column()],
                         axis=1)

    def get_mean_normalized_x_dataframe_with_time_column(self):
        """
        extract and normalize data
        """
        dataframe = self.get_x_dataframe_without_time_column()
        return pd.concat([self.get_time_dataframe(), utilities.range_scale_data(dataframe)], axis=1)

    def get_y_dataframe_without_time_column(self):
        """
        extract predictor variables.
        """
        if self.y_columns is None:
            raise Exception
        return self.dataframe.loc[:, self.y_columns]

    def get_y_dataframe_with_time_column(self):
        """
        extract predictor variables.
        """
        return pd.concat([self.get_time_dataframe(), self.get_y_dataframe_without_time_column()],
                         axis=1)

    def get_dataframe_with_time_column(self):
        """
        extract data as a new dataframe according to columns
        """
        return pd.concat([self.get_time_dataframe(),
                          self.get_x_dataframe_without_time_column(),
                          self.get_y_dataframe_without_time_column()], axis=1)

    def get_time_dataframe(self):
        """
        get time dataframe.
        """
        return self.dataframe.loc[:, "time"]
