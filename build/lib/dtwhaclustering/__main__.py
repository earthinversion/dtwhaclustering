'''
Dynamic Time Warping based Hierarchical Agglomerative Clustering
================================================================
Codes to perform Dynamic Time Warping Based Hierarchical Agglomerative Clustering of GPS data

Details
-------
This package include codes for processing the GPS displacement data including least-square modelling for trend, co-seismic jumps, 
seasonal and tidal signals. Finally, it can be used to cluster the GPS displacements based on the similarity of the waveforms. The
similarity among the waveforms will be obtained using the DTW distance.

:author: Utpal Kumar
:date: 2021/06
'''
import sys
import leastSquareModeling
import dtw_analysis

if __name__ == '__main__':
    info = \
        '''
    Dynamic Time Warping based Hierarchical Agglomerative Clustering
    ================================================================
    Codes to perform Dynamic Time Warping Based Hierarchical Agglomerative Clustering of GPS data

    Details
    -------
    This package include codes for processing the GPS displacement data including least-square modelling for trend, co-seismic jumps, 
    seasonal and tidal signals. Finally, it can be used to cluster the GPS displacements based on the similarity of the waveforms. The
    similarity among the waveforms will be obtained using the DTW distance.

    :author: Utpal Kumar
    Date: 2021/06
    '''
    sys.stdout.write(
        f"{info}\n")
