# README: Data Scaling and Model Training

In the current approach, the file `generate_features_labels.py` is used to generate the features and labels. Data scaling is handled separately in the training script, `scale_and_train.py`. In this script, the scaler is **fitted only on the training data** and then used to transform the training, validation, and test datasets. The fitted scaler is also saved for use on unseen data during inference.

**Note:** Earlier, scaling was performed on the entire dataset at once. That approach is no longer followed. The current recommended workflow is:

1. Generate datasets using `generate_features_labels.py`.
2. Train the model using `scale_and_train.py`.

For reference, this document also describes older scripts like `scale_minmax.py` and `scale_all.py` that were previously used for data scaling, as well as `train_ml_1.py` which was used for model training.

---

## Scaling in `scale_minmax.py`

### Overview
The `scale_minmax.py` script processes data from `training.txt` files across multiple time steps, separating it into three feature sets based on their physical significance:
1. **Superdroplet location offsets (`all_sdrop_1`)**: Contains offsets (`dx`, `dy`, `dz`) from the LES cell's floor and the superdroplet radius (`r`).
2. **Filtered data at the superdroplet's location (`all_sdrop_2`)**: Includes supersaturation (`s`), temperature (`T`), and velocity components (`u`, `v`, `w`).
3. **Filtered data from surrounding LES cells (`all_sdropdatales`)**: Same variables as `all_sdrop_2` but for the 27 contiguous cells around the superdroplet.

### Scaling Approach
- **Superdroplet location offsets (`all_sdrop_1`)**:
  - Offsets (`dx`, `dy`, `dz`) are scaled by dividing by the cell size (`cellsize = 51.2 / 32`), reflecting the grid's physical dimensions.
  - Radius (`r`) is scaled by dividing by a fixed maximum radius (`maxrad = 15.1`).
  - Scaling is performed as:
    ```python
    cellsize = 51.2 / 32
    maxrad = 15.1
    sdrop_1_factors = np.array([cellsize, cellsize, cellsize, maxrad])
    scaled_all_sdrop_1 = all_sdrop_1 / sdrop_1_factors
    ```

- **Filtered data (`all_sdrop_2` and `all_sdropdatales`)**:
  - Both sets contain variables (`s`, `T`, `u`, `v`, `w`) and are scaled using **min-max scaling**.
  - Data is reshaped into `(ntimesteps*nsdrops, 28, 5)` (28 = 1 superdroplet cell + 27 surrounding cells).
  - For each variable, global minimum and maximum values are computed across all data points.
  - Scaling formula:
    ```python
    scaled_value = (value - min_val) / (max_val - min_val)
    ```
  - Scaled data is reshaped back to original dimensions: `(ntimesteps*nsdrops, 5)` for `all_sdrop_2` and `(ntimesteps*nsdrops, 135)` for `all_sdropdatales`.

### Implementation
- **Data Loading**: Reads `training.txt` files, splitting data into the three feature sets.
- **Feature Concatenation**: Combines scaled features into a matrix `X` with shape `(ntimesteps*nsdrops, 144)` (4 from `all_sdrop_1` + 5 from `all_sdrop_2` + 135 from `all_sdropdatales`).
- **Saving Parameters**: Min and max values for the 5 variables are saved in `min_max_scalers.npy` for future use.

---

## Scaling in `scale_all.py`

### Overview
The `scale_all.py` script applies a uniform scaling approach to all features using scikit-learn's **StandardScaler**, which standardizes data to zero mean and unit variance.

### Scaling Approach
- **Feature Concatenation**: Combines `all_sdrop_1`, `all_sdrop_2`, and `all_sdropdatales` into a single array `all_features` with shape `(ntimesteps*nsdrops, 144)`.
- **Standardization**:
  - Fits a `StandardScaler` to `all_features`, computing mean and standard deviation per feature.
  - Transforms `all_features` to standardized values.
  - Saves the scaler as `scaler.pkl`.
- **Feature Splitting**: Optionally splits scaled features back into original subsets for clarity, though the concatenated form is used for training.

### Key Differences from `scale_minmax.py`
- Uses standardization instead of min-max scaling or fixed scaling.
- Applies a single method to all features, ignoring their physical distinctions.

---

## Training in `train_ml_1.py`

### Overview
The `train_ml_1.py` script trains a Multi-Layer Perceptron (MLP) to predict supersaturation values using scaled features from either scaling script.

### Process
1. **Data Loading**: Loads `features_hist.npy` (scaled features), `labels.npy` (supersaturation labels), and `ssdata.npy` (additional data).
2. **Data Splitting**: Splits into training (70%), validation (15%), and test (15%) sets.
3. **Model Architecture**:
   - Sequential MLP with layers:
     - 1024 neurons (ReLU) → Dropout (30%)
     - 512 neurons (ReLU) → BatchNorm → Dropout (30%)
     - 256 neurons (ReLU) → BatchNorm → Dropout (30%)
     - 64 neurons (ReLU) → BatchNorm → Dropout (30%)
     - 32 neurons (ReLU) → BatchNorm
     - 1 neuron (linear) for regression
4. **Training**:
   - Optimizer: Adam (learning rate = 0.0005)
   - Loss: Mean Squared Error (MSE)
   - Epochs: 30, Batch Size: 256
   - Monitors validation performance
5. **Evaluation**:
   - Computes MSE, MAPE, SMAPE, and R² on the test set.
   - Generates a scatter plot comparing predictions to ground truth.
6. **Saving**:
   - Model saved as `model_ml_1.keras`.
   - Metrics saved in `results.txt`.
   - Test data and predictions saved in `.npy` files.

### Notes
- GPU usage is disabled for consistent execution.
- Dropout and batch normalization enhance generalization.

---

## Training in `scale_and_train.py`
Same as `train_ml_1.py`, except it is meant to be used with `generate_features_labels.py` only. This is because the training script itself takes care of the scaling.

---


---
