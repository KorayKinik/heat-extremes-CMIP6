#### ####################################################################
#### imports
#### ####################################################################
from heatwave_utils import *

#### ####################################################################    
#### Regions
#### A class to store predefined regions and areas-of-interest
#### providsion to get region details and coordinates for the specified region ID
#### and other helpful functions in the class

#### As of the now, the regions and areas-of-interset are hard-coded
#### in future, this class can be enhanced to get regions / areas-of-interes
#### from a separate, persistent data source.

#### 11/14/2021 - As of now, the region implemented is North America and
#### the areas of interset implemented are: Mainland United States and 
#### eight sub-areas within the US.
#### ####################################################################

class Regions:
    """
    Impletemented as hard-coded values for the time being. That can be changed later to dynamic lookup.
        Available Regions:
        1 - North America. Lat (14 to 84) Lon (190 to 350)        
    """
    # individual regions
    reg_1_na = {'region_id': 1, 'region_name': 'North America', 'top_lat': 84, 'bottom_lat': 14, 'left_lon': 190, 'right_lon': 350
                , 'may_to_oct_range' : (200, 335)
                , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/Region_North_America.png'}

    # all regions
    regions = {1: reg_1_na}
    
    # individual areas-of-interest (region: NA)
    aoi_1_na_0 = {'aoi_id': 100, 'area_of_interest': 'Mainland United States', 'region_id': 1, 'region_name': 'North America'
                     , 'top_lat': 49, 'bottom_lat': 25, 'left_lon': 235, 'right_lon': 293
                     , 'may_to_oct_range' : (210, 335)
                  , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/AoI_Mainland_US.png'}

    aoi_1_na_1 = {'aoi_id': 101, 'area_of_interest': 'Upper Northeast US', 'region_id': 1, 'region_name': 'North America'
                     , 'top_lat': 46, 'bottom_lat': 39, 'left_lon': 280, 'right_lon': 293
                     , 'may_to_oct_range' : (210, 335)
                     , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/AoI_Upper_Northeast_US.png'}

    aoi_1_na_2 = {'aoi_id': 102, 'area_of_interest': 'Lower Northeast US', 'region_id': 1, 'region_name': 'North America'
                     , 'top_lat': 39, 'bottom_lat': 37, 'left_lon': 271, 'right_lon': 286
                     , 'may_to_oct_range' : (210, 335)
                     , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/AoI_Lower_Northeast_US.png'}

    aoi_1_na_3 = {'aoi_id': 103, 'area_of_interest': 'Southeast US', 'region_id': 1, 'region_name': 'North America'
                     , 'top_lat': 37, 'bottom_lat': 25, 'left_lon': 266, 'right_lon': 285
                     , 'may_to_oct_range' : (210, 335)
                     , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/AoI_Southeast_US.png'}

    aoi_1_na_4 = {'aoi_id': 104, 'area_of_interest': 'Midwest US', 'region_id': 1, 'region_name': 'North America'
                     , 'top_lat': 49, 'bottom_lat': 37, 'left_lon': 263, 'right_lon': 280
                     , 'may_to_oct_range' : (210, 335)
                     , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/AoI_Midwest_US.png'}

    aoi_1_na_5 = {'aoi_id': 105, 'area_of_interest': 'North Central US', 'region_id': 1, 'region_name': 'North America'
                     , 'top_lat': 49, 'bottom_lat': 37, 'left_lon': 245, 'right_lon': 265
                     , 'may_to_oct_range' : (210, 335)
                     , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/AoI_North_Central_US.png'}

    aoi_1_na_6 = {'aoi_id': 106, 'area_of_interest': 'South Central US', 'region_id': 1, 'region_name': 'North America'
                     , 'top_lat': 37, 'bottom_lat': 26, 'left_lon': 251, 'right_lon': 266
                     , 'may_to_oct_range' : (210, 335)
                     , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/AoI_South_Central_US.png'}

    aoi_1_na_7 = {'aoi_id': 107, 'area_of_interest': 'Northwest US', 'region_id': 1, 'region_name': 'North America'
                     , 'top_lat': 49, 'bottom_lat': 42, 'left_lon': 235, 'right_lon': 249
                     , 'may_to_oct_range' : (265, 325)
                     , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/AoI_Northwest_US.png'}

    aoi_1_na_8 = {'aoi_id': 108, 'area_of_interest': 'Southwest US', 'region_id': 1, 'region_name': 'North America'
                     , 'top_lat': 42, 'bottom_lat': 32, 'left_lon': 235, 'right_lon': 251
                     , 'may_to_oct_range' : (210, 335)
                     , 'img_url': 'https://nasanex30analysis.blob.core.windows.net/cmip6/images/AoI_Southwest_US.png'}

    # all areas-of-interest
    areas_of_interest = {100: aoi_1_na_0, 101: aoi_1_na_1, 102: aoi_1_na_2, 103: aoi_1_na_3, 104: aoi_1_na_4
                         , 105: aoi_1_na_5, 106: aoi_1_na_6, 107: aoi_1_na_7, 108: aoi_1_na_8}
    
    
    def get_region_by_ID(self, region_id):
        """
        Get region details and coordinates for the specified region ID.    
        Input:  region_id. Integer value.
        Returns: None if region_id is not found. 
                 Else, returns the dict with the following keys (value types):
                 region_id (int), region_name (str), top_lat (int), bottom_lat (int), left_lon (int), right_lon (int), img_url (str).
        """        
        return self.regions.get(region_id) 
    
    def get_all_regions(self):
        """
        Get all regions. To look up all available regions.    
        Input:  None.
        Returns: Dict of all regions.
        """        
        return self.regions
    
    def show_image(self, image_path, footnote='', **kwargs):
        if not image_path == '':
            filename = os.path.basename(image_path)
            overwrite_local_file = False if kwargs.get('overwrite_local_file') is None else kwargs.get('overwrite_local_file')
            from_azure = False if kwargs.get('from_azure') is None else kwargs.get('from_azure')
            print_msg = False if kwargs.get('print_msg') is None else kwargs.get('print_msg')
            figure_size = (8,8) if kwargs.get('figure_size') is None else kwargs.get('figure_size')
            if (isinstance(figure_size, int)):
                figure_size = (figure_size, figure_size)  # make tuple

            download_file(image_path, filename, overwrite_local_file, from_azure, print_msg)

            # read the image file in a numpy array
            img_data = plt.imread(filename)
            plt.figure(figsize = figure_size)
            plt.imshow(img_data)
            plt.axis('off')
            plt.show()

        print(footnote)
        
    
    def show_region_by_ID(self, region_id, **kwargs):
        """
        Prints out the image, if available, for the specified region ID.    
        Input:  region_id. Integer value.
             :  for kwargs parameters, see the function show_image()
        Returns: None
        """        
        # create a file-like object from the url
        img_url = self.regions.get(region_id).get('img_url')
        reg = self.regions.get(region_id)
        if img_url == '':
            print("Image URL not available for this region {}-{}".format(region_id, reg.get('region_name')))
        else:
            footnote = 'Region: {} - {}, top_lat: {}, bottom_lat: {}, left_lon: {}, right_lon: {}'.format(region_id, reg.get('region_name')
                                                                                                          , reg.get('top_lat')
                                                                                                          , reg.get('bottom_lat')
                                                                                                          , reg.get('left_lon')
                                                                                                          , reg.get('right_lon'))
            self.show_image(img_url, footnote, **kwargs)
            
            
    def get_all_areas_of_interest_by_region_ID(self, region_id):
        """
        Get details of all areas of interest for the specified region ID.    
        Input:  region_id. Integer value. 
        Returns: [] if region_id is not found. 
                 Else, returns the array of dicts with the following keys (value types):
                 'aoi_id', 'area_of_interest', 'region_id', 'region_name', 'top_lat', 'bottom_lat', 'left_lon', 'right_lon', 'img_url': ''
        """        
        return [self.areas_of_interest[k] for k in self.areas_of_interest if k//100==region_id]
    
    
    def get_area_of_interest_by_ID(self, aoi_id):
        """
        Get region details and coordinates for the specified area of interest ID.    
        Input:  aoi_id. Integer value. 
        Returns: None if aoi_id is not found. 
                 Else, returns the dict with the following keys (value types):
                 'aoi_id', 'area_of_interest', 'region_id', 'region_name', 'top_lat', 'bottom_lat', 'left_lon', 'right_lon', 'img_url': ''
        """        
        return self.areas_of_interest.get(aoi_id) 
    
    def show_areas_of_interest_by_aoi_ID(self, aoi_id, **kwargs):
        """
        Prints out the image, if available, for the specified area of interest ID.    
        Input:  aoi_id. Integer value.
             :  for kwargs parameters, see the function show_image()
        Returns: None
        """        
        # create a file-like object from the url
        aoi = self.areas_of_interest.get(aoi_id)
        if aoi is None:
            print("Area of Interest not available for this aoi_id: {}".format(aoi_id))   
        else:
            img_url = aoi.get('img_url')
            if img_url == '':
                print("Image URL not available for this area-of-interest: {}-{}\n".format(aoi_id, aoi.get('area_of_interest')))
            
            footnote = 'Area of Interest: {} - {}, top_lat: {}, bottom_lat: {}, left_lon: {}, right_lon: {}'.format(aoi_id, aoi.get('area_of_interest')
                                                                                                              , aoi.get('top_lat')
                                                                                                              , aoi.get('bottom_lat')
                                                                                                              , aoi.get('left_lon')
                                                                                                              , aoi.get('right_lon'))
            self.show_image(img_url, footnote, **kwargs)
    