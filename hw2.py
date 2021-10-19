import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.fftpack import dct
from scipy import signal
import scipy

K_MIN_DEFAULT =  1
K_MAX_DEFAULT = 10

class Training:
  def __init__(self, name, mfccs):
    self.name = name
    self.mfccs = mfccs

class Testing:
  def __init__(self, name, mfccs):
    self.name = name
    self.mfccs = mfccs

# Convert a frequency of Hz to the Mel scale
def freq_to_mel(freq):
  return 2595 * np.log10(1 + freq / 700)

# Convert a frequency of the Mel scale to Hz
def mel_to_freq(mel):
  return 700 * (10**(mel / 2595) - 1)

def average(arr):
  total = 0
  length = len(arr)
  for a in arr:
    a = np.absolute(a)
    total += a
  return total / length

# Convert m4a to wav with ffmpeg
def m4a_to_wav(filename):
  output = filename[:-4] + ".wav"
  path = "ffmpeg"
  cmd = path + " -i " + filename + " " + output + " -y"

  # If ffmpeg is not in the system PATH,
  # try the included executables in 'bin'
  if os.system(cmd) != 0:
    path = os.getcwd()
    path = os.path.join(path, "bin", "ffmpeg")
    cmd = path + " -i " + filename + " " + output + " -y"

    # Exit if included ffmpeg executable failed
    if os.system(cmd) != 0:
      print("error: ffmpeg conversion of '%s' to '%s' failed" % (filename, output))
      exit()

# Split audio signal into short frames for FFT
def frame(data, Fs):
  size = 25 / 1000   # 25ms
  step = 10 /1000    # 10ms

  # Based off data length, step, and size,
  # calculate number of frames and data per frame
  length = len(data)
  nframes = int(np.abs(length - (size*Fs)) / (step*Fs))
  data_per_frame = int(np.ceil(length / nframes))

  # Pad data with zeros until it can
  # be evenly divided into length
  while len(data) % nframes != 0:
    data = np.append(data, 0)

  # Fill frames as 2d array; each frame
  # has data_per_frame values of data
  frames = np.split(data, nframes)

  return frames

def get_mfcc(data, Fs, NFFT=512, num_filters=26):
  low_freq = 0
  high_freq = Fs / 2

  # Read data into short frames and apply hamming window to each
  frames = frame(data, Fs)
  for i in range(len(frames)):
    frames[i] = frames[i] * np.hamming(len(frames[i]))

  # Convert num_filters frequencies in Hz to the Mel scale
  mel = []
  low_mel = freq_to_mel(low_freq)
  high_mel = freq_to_mel(high_freq)
  mel_freqs = np.linspace(low_mel, high_mel, num_filters + 2)
  hz_freqs = mel_freqs

  # Convert the linear Mel freqs to Hz and convert to bins
  for i in range(len(hz_freqs)):
    hz_freqs[i] = mel_to_freq(hz_freqs[i])
  b = np.floor((NFFT + 1) * hz_freqs / Fs)

  # Map the triangular linear response 0 to 1 of each bin
  filters = np.zeros((num_filters, int(NFFT)))
  for i in range(1, num_filters+1):
    left = b[i-1]
    middle = b[i]
    right = b[i+1]

    for j in range(int(left), int(middle)):
      filters[i-1, j] = (j - left) / (middle - left)
 
    for j in range(int(middle), int(right)):
      filters[i-1, j] = (right - j) / (right - middle)

  # Apply FFT to frames to calculate power spectrum using formula:
  #          |FFT(frame)|^2
  #     P = -----------------
  #              NFFT
  P = (np.absolute(fft.rfft(frames, int(NFFT))) ** 2) / NFFT

  # Multiply filters by power spectrum to get MFCC
  mfcc = np.dot(P, filters.T)

  # Take log of each energy
  np.seterr(divide = 'ignore')
  mfcc = np.log10(mfcc)

  # Apply discrete cosine transform and only keep 2-13th of 26 coefficients
  mfcc = dct(mfcc)
  mfcc = mfcc[:,1:13]

  return mfcc
 
# Capture keystroke by waiting for drastic jump in amplitude (keystroke start)
# and filling an array with the next n secs worth of samples.
#
# The number of samples captured will be more than necessary.
#
# To detect the keystroke end, go backwards through the captured samples until
# and set them to zero until a minimum end threshold is reached.
# Setting them to zero ensures each keystroke has the same dimensions.
#
# Return a 2d array containing every key press array.
def get_keystrokes(Fs, data):
  total = 0
  start_threshold = 2000
  end_threshold = int(average(data))
  secs = .25
  samples = secs * Fs

  # Wait for an extreme amplitude jump, then
  # capture the next n secs worth of samples
  keystrokes = np.zeros((20, int(samples+2)))
  add_sample = 0
  npresses = -1
  nsamples = 0
  for i, d in enumerate(data):
    if add_sample > 0:
      keystrokes[npresses][nsamples] = d
      add_sample -= 1
      nsamples += 1
    elif abs(d) > start_threshold:
      add_sample = samples
      nsamples = 0
      npresses += 1
    elif npresses == 20:
      break

  # Detect keystroke end by iterating backwards through
  # samples and setting them to zero until a minimum
  # end threshold is reached.
  for i in range(len(keystrokes)):
    for j in range(len(keystrokes[i])-1, -1, -1):
      if keystrokes[i][j] > end_threshold:
        break
      keystrokes[i][j] =  0

  return keystrokes

def map_dataset(dataset):
  features = []
  labels = []
  for d in dataset:
    for mfcc in d.mfccs:
      for m in mfcc:
        features.append(m)
        labels.append(d.name)
  return [features, labels]

# Plot and export MFCC data for each keystroke
def export_plot(mfcc, filename):
  fig, ax = plt.subplots()
  mfcc = np.swapaxes(mfcc, 0, 1)
  cax = ax.imshow(mfcc)

  name = filename[:-4] + str(i) + ".png"
  path = os.getcwd()
  path = os.path.join(path, "plots", name)
  print("Outputting plot '%s'" % path)
  plt.title(filename)
  plt.savefig(path)
  plt.cla()

# Use K Nearest Neighbor classification to make predictions about
# testing data set compared to training data set
def knn_classify(training_data, testing_data, k):
  training_features = training_data[0]
  training_labels = training_data[1]
  testing_features = testing_data[0]
  testing_labels = testing_data[1]

  predictions = []
  for i in range(len(testing_features)):
    # Get distances between all training data points and current testing data point
    distances = np.linalg.norm(training_features - testing_features[i], axis=1)
  
    # Find the k nearest neighbors (by index) out of the distances
    knns = distances.argsort()
    knns = knns[:k]
  
    # Combine the KNNs into a prediction for current data point
    modes = []
    for knn in knns:
      label = training_labels[knn]
      modes.append(label)

    # Depending on classifer, predict based on average or mode (default=mode)
    mode = scipy.stats.mode(modes)[0]
    prediction = mode[0]
    actual = testing_labels[i]

    # Add prediction to list along with actual value
    predictions.append([prediction, actual])

  return predictions

def knn_accuracy(predictions):
  correct = 0
  for p in predictions:
    if p[0] == p[1]:
      correct += 1
  return correct / len(predictions)



##### BEGIN #####
# Try to find .wav files first to avoid relying
# on ffmpeg for .m4a to .wav conversion
basenames = ['A', 'D', 'S', 'Space']
filenames = []
for basename in basenames:
  wav = basename + '.wav'
  path = os.getcwd()
  path = os.path.join(path, wav)
  if os.path.exists(path):
    filenames.append(wav)
  else:
    filenames.append(basename + '.m4a')

NFFT = 512
num_filters = 26

# Get each file's MFCC features, split into training and
# testing sets and place into datasets keyed on filename
training_set = []
testing_set = []
for filename in filenames:
  print("File: '%s'" % filename)

  # If necessary, convert file to .wav with ffmpeg
  if filename[-4:] != ".wav":
    print("Convering '%s' to .wav" % filename)
    m4a_to_wav(filename)
    filename = filename[:-4] + ".wav"

  # Read sampling rate and data
  print("Reading file '%s'..." % filename)
  Fs, data = wavfile.read(filename)

  # Get short samples of just keystrokes
  print("Splitting audio into keystrokes...")
  keystrokes = get_keystrokes(Fs, data)

  # Plot MFCC for each keystroke, save to 'plots'
  # Split MFCC into training data (first half) and testing (last half)
  training_data = []
  testing_data = []
  print("Getting MFCC features for each keystroke...")
  for i, keystroke in enumerate(keystrokes):
    mfcc = get_mfcc(keystroke, Fs)
    export_plot(mfcc, filename)

    if i < np.floor(len(keystrokes) / 2):
      training_data.append(mfcc)
    else:
      testing_data.append(mfcc)
  plt.close('all')

  training = Training(filename[:-4], training_data)
  testing = Testing(filename[:-4], testing_data)
  training_set.append(training)
  testing_set.append(testing)
  print()


### Use K-Nearest-Neighbors (KNN) classification to classify keystrokes ###
k_min = K_MIN_DEFAULT
k_max = K_MAX_DEFAULT

# Get custom k values if provided
if len(sys.argv) == 2:
  k_min = int(sys.argv[1])
  k_max = k_min
elif len(sys.argv) == 3:
  k_min = int(sys.argv[1])
  k_max = int(sys.argv[2])
  if k_max < k_min:
    print("Error: k_max is less than k_min")
    exit()

if k_min != k_max:
  print("===Testing KNN from %d to %d===" % (k_min, k_max))

# Map both training and testing sets into array of [features, labels]
training_data = map_dataset(training_set)
testing_data  = map_dataset(testing_set)

# Test from k=k_min to k=k_max
max_accuracy = [0, 0.]
total_accuracy = 0.
for k in range(k_min, k_max+1):
  print("Testing KNN for k=%d" % k)
  predictions = knn_classify(training_data, testing_data, k)
  accuracy = knn_accuracy(predictions)
  print("Score for k=%d: %.2f percent accuracy\n" % (k, accuracy))
  if accuracy > max_accuracy[1]:
    max_accuracy = [k, accuracy]
  total_accuracy += accuracy

iterations = k_max - k_min + 1
average_accuracy = total_accuracy / iterations

# Print highest accuracy and average accuracy
print("Most accurate is k=%d with %.2f percent accuracy" % (max_accuracy[0], max_accuracy[1]))
print("Average accuracy: %.2f over %d iterations" % (average_accuracy, iterations))
