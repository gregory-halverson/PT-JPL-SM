"""
Module: generate_PTJPLSM_table_inputs.py
----------------------------------------
This module provides a function to generate the required input DataFrame for the PT-JPL-SM model, extending the original PT-JPL input requirements. It processes a calibration/validation DataFrame, computes additional variables such as hour of day, day of year (doy), Topt_C, fAPARmax, soil properties (field_capacity, wilting_point), and canopy_height_meters from spatial data sources, and appends them to the DataFrame. The function is robust to missing or problematic data, logging warnings and filling with NaN as needed.
"""
import logging

import numpy as np
import rasters as rt
from dateutil import parser
from pandas import DataFrame
from sentinel_tiles import sentinel_tiles
from solar_apparent_time import UTC_to_solar

from PTJPL import load_Topt
from PTJPL import load_fAPARmax
from soil_capacity_wilting import load_field_capacity, load_wilting_point
from gedi_canopy_height import load_canopy_height

from .model import PTJPLSM

logger = logging.getLogger(__name__)

def generate_PTJPLSM_table_inputs(
        PTJPL_inputs_from_calval_df: DataFrame,
        keep_source_columns: bool = False) -> DataFrame:
    """
    Generate a DataFrame with all required inputs for the PT-JPL-SM model.

    Parameters
    ----------
    PTJPL_inputs_from_calval_df : pandas.DataFrame
        DataFrame containing the columns: tower, lat, lon, time_UTC, albedo, elevation_km

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the original columns plus:
        - hour_of_day: int, hour of solar time at the site
        - doy: int, day of year
        - Topt_C: float, optimal temperature for photosynthesis in Celsius (from spatial data)
        - fAPARmax: float, maximum fraction of absorbed photosynthetically active radiation (from spatial data)
        - field_capacity: float, soil field capacity (from spatial data)
        - wilting_point: float, soil wilting point (from spatial data) 
        - canopy_height_meters: float, canopy height in meters (from spatial data)
        Additional columns may be added as required by the PT-JPL-SM model.

    Notes
    -----
    - This function is robust to missing or problematic spatial data; missing values are filled with np.nan.
    - The function logs progress and warnings for traceability.
    - If columns already exist in the input DataFrame, they are not overwritten.
    """
    # Copy input DataFrame to avoid modifying the original
    PTJPL_inputs_df = PTJPL_inputs_from_calval_df.copy()

    # Check which variables need to be computed
    compute_hour_of_day = "hour_of_day" not in PTJPL_inputs_df.columns
    compute_doy = "doy" not in PTJPL_inputs_df.columns
    compute_Topt_C = "Topt_C" not in PTJPL_inputs_df.columns
    compute_fAPARmax = "fAPARmax" not in PTJPL_inputs_df.columns
    compute_field_capacity = "field_capacity" not in PTJPL_inputs_df.columns
    compute_wilting_point = "wilting_point" not in PTJPL_inputs_df.columns
    compute_canopy_height = "canopy_height_meters" not in PTJPL_inputs_df.columns

    # Prepare lists to collect computed values
    hour_of_day = []
    doy = []
    Topt_C = []
    fAPARmax = []
    field_capacity_values = []
    wilting_point_values = []
    canopy_height_values = []

    # Iterate over each row to compute additional variables
    for i, input_row in PTJPL_inputs_from_calval_df.iterrows():
        # tower = input_row.tower
        tower = input_row.tower
        lat = input_row.lat
        lon = input_row.lon
        time_UTC = input_row.time_UTC
        albedo = input_row.albedo
        elevation_km = input_row.elevation_km
        logger.info(f"collecting PTJPL inputs for tower {tower} lat {lat} lon {lon} time {time_UTC} UTC")
        
        # Parse time and convert to solar time for temporal variables
        if compute_hour_of_day or compute_doy:
            time_UTC = parser.parse(str(time_UTC))
            if compute_hour_of_day:
                time_solar = UTC_to_solar(time_UTC, lon)
                hour_of_day.append(time_solar.hour)
            if compute_doy:
                doy.append(time_UTC.timetuple().tm_yday)
        
        # Skip spatial processing if no spatial variables need to be computed
        if not any([compute_Topt_C, compute_fAPARmax, compute_field_capacity, compute_wilting_point, compute_canopy_height]):
            continue
        
        try:
            # Get MGRS tile and grid for spatial data extraction
            tile = sentinel_tiles.toMGRS(lat, lon)[:5]
            tile_grid = sentinel_tiles.grid(tile=tile, cell_size=70)
        except Exception as e:
            logger.warning(e)
            if compute_Topt_C:
                Topt_C.append(np.nan)
            if compute_fAPARmax:
                fAPARmax.append(np.nan)
            if compute_field_capacity:
                field_capacity_values.append(np.nan)
            if compute_wilting_point:
                wilting_point_values.append(np.nan)
            if compute_canopy_height:
                canopy_height_values.append(np.nan)
            continue

        rows, cols = tile_grid.shape
        # Find the grid cell containing the point
        row, col = tile_grid.index_point(rt.Point(lon, lat))
        # Extract a 3x3 neighborhood around the point for robust statistics
        geometry = tile_grid[max(0, row - 1):min(row + 2, rows - 1),
                             max(0, col - 1):min(col + 2, cols - 1)]

        # Compute Topt_C if needed
        if compute_Topt_C:
            try:
                logger.info("generating Topt_C")
                Topt_value = np.nanmedian(load_Topt(geometry=geometry))
                print(f"Topt_C: {Topt_value}")
                Topt_C.append(Topt_value)
            except Exception as e:
                Topt_C.append(np.nan)
                logger.exception(e)
        
        # Compute fAPARmax if needed
        if compute_fAPARmax:
            try:
                logger.info("generating fAPARmax")
                fAPARmax_value = np.nanmedian(load_fAPARmax(geometry=geometry))
                print(f"fAPARmax: {fAPARmax_value}")
                fAPARmax.append(fAPARmax_value)
            except Exception as e:
                fAPARmax.append(np.nan)
                logger.exception(e)
        
        # Compute field_capacity if needed
        if compute_field_capacity:
            try:
                logger.info("generating field_capacity")
                field_capacity_value = np.nanmedian(load_field_capacity(geometry=geometry))
                print(f"field_capacity: {field_capacity_value}")
                field_capacity_values.append(field_capacity_value)
            except Exception as e:
                field_capacity_values.append(np.nan)
                logger.exception(e)
        
        # Compute wilting_point if needed
        if compute_wilting_point:
            try:
                logger.info("generating wilting_point")
                wilting_point_value = np.nanmedian(load_wilting_point(geometry=geometry))
                print(f"wilting_point: {wilting_point_value}")
                wilting_point_values.append(wilting_point_value)
            except Exception as e:
                wilting_point_values.append(np.nan)
                logger.exception(e)
        
        # Compute canopy_height_meters if needed
        if compute_canopy_height:
            try:
                logger.info("generating canopy_height_meters")
                canopy_height_value = np.nanmedian(load_canopy_height(geometry=geometry))
                print(f"canopy_height_meters: {canopy_height_value}")
                canopy_height_values.append(canopy_height_value)
            except Exception as e:
                canopy_height_values.append(np.nan)
                logger.exception(e)
    
    # Add computed columns to DataFrame
    if compute_hour_of_day:
        PTJPL_inputs_df["hour_of_day"] = hour_of_day

    if compute_doy:
        PTJPL_inputs_df["doy"] = doy
    
    if compute_Topt_C:
        PTJPL_inputs_df["Topt_C"] = Topt_C
    
    if compute_fAPARmax:
        PTJPL_inputs_df["fAPARmax"] = fAPARmax
    
    if compute_field_capacity:
        PTJPL_inputs_df["field_capacity"] = field_capacity_values
    
    if compute_wilting_point:
        PTJPL_inputs_df["wilting_point"] = wilting_point_values
    
    if compute_canopy_height:
        PTJPL_inputs_df["canopy_height_meters"] = canopy_height_values
    
    # Rename temperature column if needed for model compatibility
    if "Ta" in PTJPL_inputs_df and "Ta_C" not in PTJPL_inputs_df:
        PTJPL_inputs_df.rename({"Ta": "Ta_C"}, inplace=True)
    
    if not keep_source_columns:
        columns = [
            "tower",
            "lat",
            "lon",
            "time_UTC",
            "albedo",
            "elevation_km",
            "hour_of_day",
            "doy",
            "Topt_C",
            "fAPARmax",
            "field_capacity",
            "wilting_point",
            "canopy_height_meters",
            "Ta_C",
            "NDVI",
            "ST_C",
            "RH",
            "SM",
            "geometry"
        ]
        
        PTJPL_inputs_df = PTJPL_inputs_df[columns]

    return PTJPL_inputs_df
