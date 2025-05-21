# Data Format Specification for WristMET Calculator API

This document outlines the expected format for uploading sensor data to the [WristMET Calculator](https://wristmetcalculator.fsm.northwestern.edu/). You must provide three separate CSV files, each corresponding to a sensor modality:

- `accelerometer.csv`
- `gyroscope.csv`
- `actigraph.csv`

All timestamps should be synchronized across modalities whenever possible.

## 1. `accelerometer.csv`

### Format:
- **File type**: CSV
- **Sampling rate**: 50–100 Hz
- **Columns**:
  - `Time`: Epoch timestamp in milliseconds
  - `accX`: Acceleration in X-axis (in g)
  - `accY`: Acceleration in Y-axis (in g)
  - `accZ`: Acceleration in Z-axis (in g)

### Example:
| Time           | accX     | accY     | accZ     |
|----------------|----------|----------|----------|
| 1581524000000  | -1.5646  | 9.1792   | 7.4825   |
| 1581524000100  | 0.1506   | 6.6089   | 3.5706   |

## 2. `gyroscope.csv`

### Format:
- **File type**: CSV
- **Sampling rate**: Should match accelerometer (50–100 Hz)
- **Columns**:
  - `Time`: Epoch timestamp in milliseconds
  - `rotX`: Angular velocity around X-axis (deg/s)
  - `rotY`: Angular velocity around Y-axis
  - `rotZ`: Angular velocity around Z-axis

### Example:
| Time           | rotX     | rotY     | rotZ     |
|----------------|----------|----------|----------|
| 1581524000000  | 0.9106   | 0.5551   | -1.3516  |
| 1581524000100  | -0.9602  | 1.1097   | -2.7748  |


## 3. `actigraph.csv`

### Format:
- **File type**: CSV exported from ActiGraph software (e.g., `.agd`)
- **Sampling rate**: Aggregated per-minute epochs
- **Important Notes**:
  - The first row contains metadata and must be removed or skipped by the API
  - Columns may include time, axis counts, vector magnitude (vm), steps, inclinometer, calories, and METs

### Expected Columns (simplified view):
- `date`
- `epoch`
- `axis1`, `axis2`, `axis3`
- `vm` (vector magnitude)
- `steps`
- `lux`
- `inclinometer` states
- `kcals`
- `MET rate`

### Example:
| date            | epoch | axis1 | axis2 | axis3 | vm   | steps | kcals | MET rate |
|------------------|--------|--------|--------|--------|--------|--------|--------|-----------|
| 2/12/2020 9:29AM | 650    | 2405   | 618    | 2567   | 25     | 0      | 7.194  | 1         |

---

## File Naming Convention

- The filenames **must exactly** match the required names:
  - `accelerometer.csv`
  - `gyroscope.csv`
  - `actigraph.csv`

