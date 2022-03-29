from __future__ import division # ensures no rounding errors from division involving integers
from math import * # enables use of pi, trig functions, and more.
import pandas as pd # gives us the dataframe concept
pd.options.display.max_columns = 50
pd.options.display.max_rows = 9
import streamlit as st

from pyhdf import SD
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import base64

st.set_page_config(
page_title="CAP Solar",
page_icon="🚩"
)
# set background, use base64 to read local file
def get_base64_of_bin_file(bin_file):
    """
    function to read png file 
    ----------
    bin_file: png -> the background image in local folder
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    """
    function to display png as bg
    ----------
    png_file: png -> the background image in local folder
    """
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return



set_png_as_page_bg('solar3.gif')
  
st.markdown("<h1 style ='color:pink; text_align:center;font-family:times new roman;font-weight: bold;font-size:20pt;'>Impact of Aerosols in Solar Power Generation </h1>", unsafe_allow_html=True)  

file_name="MYD04_L2.A2022051.0000.061.2022051163849.hdf"
hdf=SD.SD(file_name)
dataFields=dict([(1,'Deep_Blue_Aerosol_Optical_Depth_550_Land'),(2,'AOD_550_Dark_Target_Deep_Blue_Combined'),(3,'AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag')])
SDS_NAME=dataFields[2] # The name of the sds to read

# Get lat and lon info
lat = hdf.select('Latitude')
latitude = lat[:,:]
min_lat=latitude.min()
max_lat=latitude.max()
lon = hdf.select('Longitude')
longitude = lon[:,:]
min_lon=longitude.min()
max_lon=longitude.max()
		
#get SDS
try:
	sds=hdf.select(SDS_NAME)
except:
	print('Sorry, your MODIS hdf file does not contain the SDS:',SDS_NAME,'. Please try again with the correct file type.')

#get scale factor and fill value for data field
attributes=sds.attributes()
scale_factor=attributes['scale_factor']
fillvalue=attributes['_FillValue']
		
#get SDS data
data=sds.get()
#Print the range of latitude and longitude found in the file, then ask for a lat and lon
#st.write('The range of latitude in this file is: ',min_lat,' to ',max_lat, 'degrees \nThe range of longitude in this file is: ',min_lon, ' to ',max_lon,' degrees')
st.markdown("<h5 style ='color:white; text_align:center;font-family:times new roman;font-weight: bold;font-size:12pt;'>Please enter the latitude (between 14.599766 to 35.4595 degrees) (Deg. N):  </h5>", unsafe_allow_html=True)  
user_lat=st.text_input('.')
st.markdown("<h5 style ='color:white; text_align:center;font-family:times new roman;font-weight: bold;font-size:12pt;'>Please enter the longitude (between -174.43806 to -147.60152 degrees) (Deg. E): </h5>", unsafe_allow_html=True)  
user_lon=st.text_input(' ')


if st.button("Predict"):
    user_lat=float(user_lat)
    user_lon=float(user_lon)
    df_map = pd.DataFrame(
	 np.random.randn(1000, 2) / [50, 50] + [user_lat,user_lon],
    columns=['lat', 'lon'])
    st.markdown("<h1 style='text-align: left; font-weight:bold;color:black;background-color:white;font-size:11pt;'> Selected Location </h1>",unsafe_allow_html=True)
    st.map(df_map)
    #calculation to find nearest point in data to entered location (haversine formula)
    R=6371000#radius of the earth in meters
    lat1=np.radians(user_lat)
    lat2=np.radians(latitude)
    delta_lat=np.radians(latitude-user_lat)
    delta_lon=np.radians(longitude-user_lon)
    a=(np.sin(delta_lat/2))*(np.sin(delta_lat/2))+(np.cos(lat1))*(np.cos(lat2))*(np.sin(delta_lon/2))*(np.sin(delta_lon/2))
    c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    d=R*c
    #gets (and then prints) the x,y location of the nearest point in data to entered location, accounting for no data values
    x,y=np.unravel_index(d.argmin(),d.shape)
    print('\nThe nearest pixel to your entered location is at: \nLatitude:',latitude[x,y],' Longitude:',longitude[x,y])
    if data[x,y]==fillvalue:
        	st.write('The value of ',SDS_NAME,'at this pixel is',fillvalue,',(No Value)\n')
        	AOD500nm=fillvalue
    else:
        	st.write('The value of ', SDS_NAME,'at this pixel is ',round(data[x,y]*scale_factor,3))
        	AOD500nm=round(data[x,y]*scale_factor,3)

    ### User Inputs
    
    phi = user_lat
    longitude = user_lon
    tz = -7
    P_mb = 840
    Ozone_cm = 0.3
    H20_cm = 1.5
    #AOD500nm = 0.1
    AOD380nm = 0.15
    Taua = 0.08
    Ba = 0.85
    albedo = 0.2
    
    G_sc = 1367 # W/m^2
    std_mer = longitude-longitude%15+15 # This Standard Meridian calculation is only a guide!! 
                                        # Please double check this value for your location!
    
    ### Day of the Year Column
    
    n = range(1,366) # julian day of the year
    n_hrly = list(pd.Series(n).repeat(24)) # julian day numbers repeated hourly to create 8760 datapoints in dataset
    
    ds = pd.DataFrame(n_hrly, columns=['DOY']) # create dataframe with julian days 
    
    ### Hr of the Day Column
    
    ds['HR'] = [(hr)%24 for hr in ds.index.tolist()] # append dataframe with hr of the day for each day
    
    ### Extraterrestrial Radiation
    
    def etr(n):
        return G_sc*(1.00011+0.034221*cos(2*pi*(n-1)/365)+0.00128*sin(2*pi*(n-1)/365)+0.000719*cos(2*(2*pi*(n-1)/365))+0.000077*sin(2*(2*pi*(n-1)/365)))
    
    ds['ETR'] = [etr(n) for n in ds['DOY']] # append dataframe with etr for day
    
    ### Intermediate Parameters
    
    ds['Dangle'] = [2*pi*(n-1)/365 for n in ds['DOY']]

    def decl(Dangle):
        return (0.006918-0.399912*cos(Dangle)+0.070257*sin(Dangle)-0.006758*cos(2*Dangle)+0.000907*sin(2*Dangle)-0.002697*cos(3*Dangle)+0.00148*sin(3*Dangle))*(180/pi)
    ds['DEC'] = [decl(Dangle) for Dangle in ds['Dangle']]
    
    def eqtime(Dangle):
        return (0.0000075+0.001868*cos(Dangle)-0.032077*sin(Dangle)-0.014615*cos(2*Dangle)-0.040849*sin(2*Dangle))*229.18
    ds['EQT'] = [eqtime(Dangle) for Dangle in ds['Dangle']]
    
    def omega(hr, eqt):
        return 15*(hr-12.5) + longitude - tz*15 + eqt/4
    ds['Hour Angle'] = [omega(hr, eqt) for hr, eqt in zip(ds['HR'],ds['EQT'])]
    
    def zen(dec, hr_ang):
        return acos(cos(dec/(180/pi))*cos(phi/(180/pi))*cos(hr_ang/(180/pi))+sin(dec/(180/pi))*sin(phi/(180/pi)))*(180/pi)
    ds['Zenith Ang'] = [zen(dec, hr_ang) for dec, hr_ang in zip(ds['DEC'],ds['Hour Angle'])]
    
    def airmass(zenang):
        if zenang < 89:
            return 1/(cos(zenang/(180/pi))+0.15/(93.885-zenang)**1.25)
        else:
            return 0
    ds['Air Mass'] = [airmass(zenang) for zenang in ds['Zenith Ang']]
    
    ### Intermediate Results
    
    def T_rayleigh(airmass):
        if airmass > 0:
            return exp(-0.0903*(P_mb*airmass/1013)**0.84*(1+P_mb*airmass/1013-(P_mb*airmass/1013)**1.01))
        else:
            return 0
    ds['T rayleigh'] = [T_rayleigh(airmass) for airmass in ds['Air Mass']]
    
    def T_ozone(airmass):
        if airmass > 0:
            return 1-0.1611*(Ozone_cm*airmass)*(1+139.48*(Ozone_cm*airmass))**-0.3034-0.002715*(Ozone_cm*airmass)/(1+0.044*(Ozone_cm*airmass)+0.0003*(Ozone_cm*airmass)**2)
        else:
            return 0
    ds['T ozone'] = [T_ozone(airmass) for airmass in ds['Air Mass']]
    
    def T_gasses(airmass):
        if airmass > 0:
            return exp(-0.0127*(airmass*P_mb/1013)**0.26)
        else:
            return 0
    ds['T gases'] = [T_gasses(airmass) for airmass in ds['Air Mass']]
    
    def T_water(airmass):
        if airmass > 0:
            return 1-2.4959*airmass*H20_cm/((1+79.034*H20_cm*airmass)**0.6828+6.385*H20_cm*airmass)
        else:
            return 0
    ds['T water'] = [T_water(airmass) for airmass in ds['Air Mass']]
    
    def T_aerosol(airmass):
        if airmass > 0:
            return exp(-(Taua**0.873)*(1+Taua-Taua**0.7088)*airmass**0.9108)
        else:
            return 0
    ds['T aerosol'] = [T_aerosol(airmass) for airmass in ds['Air Mass']]
    
    def taa(airmass, taerosol):
        if airmass > 0:
            return 1-0.1*(1-airmass+airmass**1.06)*(1-taerosol)
        else:
            return 0
    ds['TAA'] = [taa(airmass, taerosol) for airmass, taerosol in zip(ds['Air Mass'],ds['T aerosol'])]
    
    def rs(airmass, taerosol, taa):
        if airmass > 0:
            return 0.0685+(1-Ba)*(1-taerosol/taa)
        else:
            return 0
    ds['rs'] = [rs(airmass, taerosol, taa) for airmass, taerosol, taa in zip(ds['Air Mass'],ds['T aerosol'],ds['TAA'])]
    
    def Id(airmass, etr, taerosol, twater, tgases, tozone, trayleigh):
        if airmass > 0:
            return 0.9662*etr*taerosol*twater*tgases*tozone*trayleigh
        else:
            return 0
    ds['Id'] = [Id(airmass, etr, taerosol, twater, tgases, tozone, trayleigh) for airmass, etr, taerosol, twater, tgases, tozone, trayleigh in zip(ds['Air Mass'],ds['ETR'],ds['T aerosol'],ds['T water'],ds['T gases'],ds['T ozone'],ds['T rayleigh'])]
    
    def idnh(zenang, Id):
        if zenang < 90:
            return Id*cos(zenang/(180/pi))
        else:
            return 0
    ds['IdnH'] = [idnh(zenang, Id) for zenang, Id in zip(ds['Zenith Ang'],ds['Id'])]
    
    def ias(airmass, etr, zenang, tozone, tgases, twater, taa, trayleigh, taerosol):
        if airmass > 0:
            return etr*cos(zenang/(180/pi))*0.79*tozone*tgases*twater*taa*(0.5*(1-trayleigh)+Ba*(1-(taerosol/taa)))/(1-airmass+(airmass)**1.02)
        else:
            return 0
    ds['Ias'] = [ias(airmass, etr, zenang, tozone, tgases, twater, taa, trayleigh, taerosol) for airmass, etr, zenang, tozone, tgases, twater, taa, trayleigh, taerosol in zip(ds['Air Mass'],ds['ETR'],ds['Zenith Ang'],ds['T ozone'],ds['T gases'],ds['T water'],ds['TAA'],ds['T rayleigh'],ds['T aerosol'])]
    
    def gh(airmass, idnh, ias, rs):
        if airmass > 0:
            return (idnh+ias)/(1-albedo*rs)
        else:
            return 0
    ds['GH'] = [gh(airmass, idnh, ias, rs) for airmass, idnh, ias, rs in zip(ds['Air Mass'],ds['IdnH'],ds['Ias'],ds['rs'])]
    
    ### Decimal Time
    
    def dectime(doy, hr):
        return doy+(hr-0.5)/24
    ds['Decimal Time'] = [dectime(doy, hr) for doy, hr in zip(ds['DOY'],ds['HR'])]
    
    ### Model Results (W/m^2)
    
    ds['Direct Beam'] = ds['Id']
    
    ds['Direct Hz'] = ds['IdnH']
    
    ds['Global Hz'] = ds['GH']
    
    ds['Dif Hz'] = ds['Global Hz']-ds['Direct Hz']
    
    ds[11:15]
    
    
    pylab.rcParams['figure.figsize'] = 16, 6  # this sets the default image size for this session
    
    ax = ds[ds['DOY']==212].plot('HR',['Global Hz','Direct Hz','Dif Hz'],title='Bird Clear Sky Model Results')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Irradiance W/m^2")
    majorx = ax.set_xticks(range(0,25,1))
    majory = ax.set_yticks(range(0,1001,200))
    print(ds[11:15])
    
    min_index=ds['Direct Beam'].argmin()
    print(min_index)
    
    solar_irradiance=ds['Dif Hz'].mean()
    st.success("OUTPUT OF LIGHT ENERGY FROM THE SUN {} W/m2".format(round(solar_irradiance,2)))
	#1 Unit “kilowatt-hour (kWh)” Cost:₹9
    st.write("1 Unit Cost:₹9")
    st.warning("Electricity Cost per Day: ₹ {}".format(round((solar_irradiance*9*24)/1000,2)))
    st.info("Electricity Cost per Month: ₹ {}".format(round((solar_irradiance*9*720)/1000,2)))
	
