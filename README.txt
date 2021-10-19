Python version: 3.7.9
Required libraries: matplotlib, numpy, scipy
`pip3 install matplotlib numpy scipy`


To run:

`python3 hw2.py`
will use default k_min and k_max values to test KNN over a short range

OR

`python3 hw2.py k`
insert a number in place of 'k' to test that specific k value for KNN

OR

`python3 hw2.py k_min k_max`
insert a number in place of 'k_min' and another for 'k_max' to test KNN over custom range


ffmpeg is used to convert .m4a files to .wav.
The .wav files should be included already, and any .wav files found will be used instead of the .m4a to avoid needing to convert with ffmpeg.
If ffmpeg is in the system's PATH, it will be used to convert. Otherwise the included Windows .exe and Linux binary in 'bin' will be used.

Also, the keystroke end detection works by setting amplitudes to zero in reverse order until a minimum threshold is reached. This will cause the exported plots to have some white space on the end.
