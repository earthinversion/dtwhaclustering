# Dynamic Time Warping based Hierarchical Agglomerative Clustering

Codes to perform Dynamic Time Warping Based Hierarchical Agglomerative Clustering of GPS data

## Details

This package include codes for processing the GPS displacement data including least-square modelling for trend, co-seismic jumps, 
seasonal and tidal signals. Finally, it can be used to cluster the GPS displacements based on the similarity of the waveforms. The
similarity among the waveforms will be obtained using the DTW distance.


## Usage
### Least-squares modeling

![Load Pickle Data into Pandas DataFrame](https://raw.githubusercontent.com/earthinversion/DTW-based-Hierarchical-Clustering/master/images/load_data.png?token=ADNOWX7G3OSZIVAAFBM2DADAZSOG2)

```
from dtwhaclustering.leastSquareModeling import lsqmodeling
final_dU, final_dN, final_dE = lsqmodeling(dUU, dNN, dEE,stnlocfile="helper_files/stn_loc.txt",  plot_results=True, remove_trend=False, remove_seasonality=True, remove_jumps=False)
```

![LSQ Model](https://raw.githubusercontent.com/earthinversion/DTW-based-Hierarchical-Clustering/master/images/time_series_SLNP_U.png?token=ADNOWX6QYC7CK3FDFECN4X3AZSOIW)

## License
© 2021 Utpal Kumar

Licensed under the Apache License, Version 2.0