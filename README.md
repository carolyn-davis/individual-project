# individual-project
## Project Summary
For this project I sought to identify and predict upon key drivers of hurricanes in the Caribbean region of the Atlantic Ocean that are supported with various machine learning models. The data for this project was acquired from [Kaggle](https://www.kaggle.com/padamaxnassetti/hurricanes-caribbean/data). The data known as Hurricanes and Typhoons 1851-2015 originates from public data repositories provided by National Hurricane Center(NHC). Of this dataset the Atlantic was chosen for the development of this project. With goal of identifying the specified drivers, it was also a goal of this project to perform all stages stages of the Data Science pipeline, providing key observation, takeaways, and thorough documentation for all actions takes through the course of the project. Further documentation for this project can be found in the documentation provided alongside the dataset.

## Project Goal:
- Utilized time series methodologies to identify key drivers of hurricane occurence (the target)
- Create modules that retrieve and prepare the data with use of various functions 
- Provide a step by step walkthrough, documnenting thoroughly for project recreation
- Develop and produce atleast 3 models supporting future predictions on the data
- Ensure steps are well documented to ensure reproduction of project 

## Deliverables:
- Github Repo w/ Final Notebook and README

## Project Planning:
#### Prepare Stage Planning:
   - Goal:  Leave this stage a wranglecarribean.py file saved to git repo
                Pair along with steps for other users to recreate 
    Summary:
    - Column data types are appropriate for the data they contain
    - Missing values are investigated and handled
    - Outliers are investigated and handled

   - Prepare Stage Checklist:
        1. change values columns
        2. investigate missing values
        3. investigate any possible outliers
        4. If outliers, should they be included/excluded?
        5. Rename the columns for readibility
        6. Unique value counts for future selection
       
#### Explore Stage Planning:
   - Goal: Identify the key driver of the hurricanes by exploring drivers based on their association with feature picked 
        -Additionally define and visualize those paatterns
    Summary:
    - Investigate the interaction between independent variables and the target variable is explored using visualization and statistical testing.
    
  - Explore Stage Checklist:
        1. Initial Hypothesises documented, Is there a relationship between max_wind and hurricane occurence in the Carribean?
        2. Plotting/Visualizations: histographs, barplots
        3. Statistical Testing :  Lag/Autocorrelation
        4. Visual: Lag/Autocorrelation Plots
        5. Establish Observations/Takeaways after each visualization and statistical test
        
#### Modeling Stage Planning:
   - Goal/Summary:
    Develop atleast 3 model and evaluate their individual performance. One model is the distinct combination of algorithm, hyperparameters, and features.
    - Linear and Nonlinear models were created and tested for this project
    - Models created and tested: 'Last Observed Value, Simple Average, Moving Average, and Prophet'
        
  - Modeling Stage Checklist:
        1. Establish the Scaled/Unscaled** data for modeling
        2. Set features
        3. Establish the baseline
        4. Run models on Train
        5. Run models on Validate
        6. Run models on Test
        7. Observe Results in comparison to the baseline

#### Deliver Stage Planning:
   - Goal/Summary: You are expected to deliver a github repository with the following contents:
        - A clearly named **final notebook**. This notebook will be what you present and should contain plenty of markdown documentation and cleaned up code.
        - A README that explains what the project is, how to reproduce you work, and your notes from project planning.

    
   - Deliver Stage Checklist:
        1. github repo called individual-project
        2. a final jupyter notebook for walkthrough of project steps
        3. ReadMe.md all about the project and how to recreate it
        4. acquire.py file for retrieving Zillow data
        5. wranglecarribean.py file that retrieves your prepared/cleaned data
        6. data dictionary

             ### DATA DICTIONARY FOR THIS PROJECT 
- C – Closest approach to a coast, not followed by a landfall
- G – Genesis
- I – An intensity peak in terms of both pressure and wind
- L – Landfall (center of system crossing a coastline)
- P – Minimum in central pressure
- R – Provides additional detail on the intensity of the cyclone when rapid changes are underway
- S – Change of status of the system
- T – Provides additional detail on the track (position) of the cyclone
- W – Maximum sustained wind speed


status_of_system — The type of storm. Options are:

- TD – Tropical cyclone of tropical depression intensity (< 34 knots)
- TS – Tropical cyclone of tropical storm intensity (34-63 knots)
- HU – Tropical cyclone of hurricane intensity (> 64 knots)
- EX – Extratropical cyclone (of any intensity)
- SD – Subtropical cyclone of subtropical depression intensity (< 34 knots)
- SS – Subtropical cyclone of subtropical storm intensity (> 34 knots)
- LO – A low that is neither a tropical cyclone, a subtropical cyclone, nor an extratropical cyclone (of any intensity)
- WV – Tropical Wave (of any intensity)
- DB – Disturbance (of any intensity)

- latitude — The latitude (y positional component).
- longitude — The longitude (x positional component).
- maximum_sustained_wind_knots — The maximum 1-min average wind associated with the tropical cyclone at an elevation of 10 m with an unobstructed exposure, in knots (kt).
- maximum_pressure — The central atmospheric pressure of the hurricane.
- 34_kt_ne
- 34_kt_se
- 34_kt_sw
- 34_kt_nw
- 50_kt_ne
- 50_kt_se
- 50_kt_sw
- 50_kt_nw
- 64_kt_ne
- 64_kt_se
- 64_kt_sw
- 64_kt_nw 

— This entry and those above it together indicate the boundaries of the storm's radius of maximum wind. 34 knots is considered tropical storm force winds, 50 knots is considered storm force winds, and 64 knots is considered hurricane force winds (source). These measurements provide the distance (in nautical miles) from the eye of the storm (its latitude, longitude entry) in which winds of the given force can be expected. This information is only available for observations since 2004.

- [Original Data Description ](https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atlantic.pdf)
Cyclone number: In HURDAT2, the order cyclones appear in the file is determined by the date/time of the first tropical or subtropical cyclone record in the best track. This sequence may or may not correspond to the ATCF cyclone number. For example, the 2011 unnamed tropical storm AL20 which formed on 1 September, is sequenced here between AL12 (Katia – formed on 29 Aug) and AL13 (Lee – formed on 2 September). This mismatch between ATCF cyclone number and the HURDAT2 sequencing can occur if post-storm analysis alters the relative genesis times between two cyclones. In addition, in 2011 it became practice to assign operationally unnamed cyclones ATCF numbers from the end of the list, rather than insert them in sequence and alter the ATCF numbers of cyclones previously assigned.

- Name: Tropical cyclones were not formally named before 1950 and are thus referred to as “UNNAMED” in the database. Systems that were added into the database after the season (such as AL20 in 2011) also are considered “UNNAMED”. Non-developing tropical depressions formally were given names (actually numbers, such as “TEN”) that were included into the ATCF b-decks starting in 2003. Non-developing tropical depressions before this year are also referred to as “UNNAMED”.

- Record identifier: This code is used to identify records that correspond to landfalls or to indicate the reason for inclusion of a record not at the standard synoptic times (0000, 0600, 1200, and 1800 UTC). For the years 1851-1955, 1969’s Camille, and 1991 onward, all continental United States landfalls are marked, while international landfalls are only marked from 1951 to 1955 and 1991 onward. The landfall identifier (L) is the only identifier that will appear with a standard synoptic time record. The remaining identifiers (see table above) are only used with asynoptic records to indicate the reason for their inclusion. Inclusion of asynoptic data is at the discretion of the Hurricane Specialist who performed the post-storm analysis; standards for inclusion or non-inclusion have varied over time. Identification of asynoptic peaks in intensity (either wind or pressure) may represent either system’s lifetime peak or a secondary peak

- Time: Nearly all HURDAT2 records correspond to the synoptic times of 0000, 0600, 1200, and 1800. Recording best track data to the nearest minute became available within the b-decks beginning in 1991 and some tropical cyclones since that year have the landfall best track to the nearest minute.

- Status: Tropical cyclones with an ending tropical depression status (the dissipating stage) were first used in the best track beginning in 1871, primarily for systems weakening over land. Tropical cyclones with beginning tropical depression (the formation stage) were first included in the best track beginning in 1882. Subtropical depression and subtropical storm status were first used beginning in 1968 at the advent of routine satellite imagery for the Atlantic basin. The low status – first used in 1987 - is for cyclones that are not tropical cyclone or subtropical cyclones, nor extratropical cyclones. These typically are assigned at the beginning of a system’s lifecycle and/or at the end of a system’s lifecycle. The tropical wave status – first used in 1981 - is almost exclusively for cyclones that degenerate into an open trough for a time, but then redevelop later in time into a tropical cyclone (for example, AL10-DENNIS in 1981 between 13 and 15 August). The disturbance status is similar to tropical wave and was first used in 1980. It should be noted that for tropical wave and disturbance status the location given is the approximate position of the lower tropospheric vorticity center, as the surface center no longer exists for these stages.

- Maximum sustained surface wind: This is defined as the maximum 1-min average wind associated with the tropical cyclone at an elevation of 10 m with an unobstructed exposure. Values are given to the nearest 10 kt for the years 1851 through 1885 and to the nearest 5 kt from 1886 onward. A value is assigned for every cyclone at every best track time. Note that the non-developing tropical depressions of 1967 did not have intensities assigned to them in the b-decks. These are indicated as “-99” currently, but will be revised and assigned an intensity when the Atlantic hurricane database reanalysis project (Hagen et al. 2012) reaches that hurricane season.

- Central Pressure: These values are given to the nearest millibar. Originally, central pressure best track values were only included if there was a specific observation that could be used explicitly. Missing central pressure values are noted as “-999”. Beginning in 1979, central pressures have been analyzed and included for every best track entry, even if there was not a specific in-situ measurement available.

- Wind Radii – These values have been best tracked since 2004 and are thus available here from that year forward with a resolution to the nearest 5 nm. Best tracks of the wind radii have not been done before 2004 and are listed as “-999” to denote missing data. Note that occasionally when there is a non-synoptic time best track entry included for either landfall or peak intensity, that the wind radii best tracks were not provided. These instances are also denoted with a “-999” in the database.

- General Notes: The database goes back to 1851, but it is far from being complete and accurate for the entire century and a half. Uncertainty estimates of the best track parameters available for are available for various era in Landsea et al. (2012), Hagen et al. (2012), Torn and Snyder (2012), and Landsea and Franklin (2013). Moreover, as one goes back further in time in addition to larger uncertainties, biases become more pronounced as well with tropical cyclone frequencies being underreported and the tropical cyclone intensities being underanalyzed. That is, some storms were missed and many intensities are too low in the pre-aircraft reconnaissance era (1944 for the western half of the basin) and in the pre-satellite era (late-1960s for the entire basin). Even in the last decade or two, new technologies affect the best tracks in a non-trivial way because of our generally improving ability to observe the frequency, intensity, and size of tropical cyclones. See Vecchi and Knutson (2008), Landsea et al. (2010), Vecchi and Knutson (2012), Uhlhorn and Nolan (2012) on methods that have been determined to address some of the undersampling issues that arise in monitoring these mesoscale, oceanic phenomenon.
