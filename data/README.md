# Data

Place event stream files here. Large binary files are excluded from the repository via `.gitignore`.

## Annotations

`data/annotations/` contains motion-state labels manually annotated by the authors.
Annotations follow the Prophesee bounding box format (structured NumPy array with
fields `t, x, y, w, h, class_id, class_confidence, track_id`). The `class_id` field
encodes both object type and motion state:

| `class_id` | Label |
|---|---|
| 0 | person — static |
| 1 | vehicle — static |
| 2 | person — moving |
| 3 | vehicle — moving |

## MVSEC

Download the `outdoor_day2` sequence from [daniilidis-group.github.io/mvsec](https://daniilidis-group.github.io/mvsec/).

MVSEC distributes data as `*_data.hdf5` files containing many sensor streams.
The GeoIMO pipeline expects events in the Prophesee `.dat` format, so convert
the HDF5 first with the provided helper. `h5py` is bundled in the container, so
you can run it directly:

```bash
apptainer exec geoimo.sif python data/MVSEC_prophesee_converter.py \
    --input_hdf5 /path/to/outdoor_day2_data.hdf5 \
    --output_dat data/outdoor_day2_events.dat
```

> **Note:** Apptainer/Singularity only auto-mounts your home directory and the
> current working directory. If the HDF5 lives elsewhere, bind-mount its folder,
> e.g. `apptainer exec --bind /data/MVSEC:/mvsec geoimo.sif python ... --input_hdf5 /mvsec/outdoor_day2_data.hdf5 ...`.

Without the container, install `h5py` first (`pip install h5py numpy`) and run
the same command with plain `python`.

The original MVSEC dataset does not include bounding box annotations. Bounding boxes
were generated from the dataset's greyscale images using a YOLO-family object detector
and then manually annotated with motion-state labels following the class scheme above.
These annotations are provided in `data/annotations/outdoor_day2_motion_labels.npy`.

Example run:
```bash
DATASET=mvsec \
EVENT_PATH=data/outdoor_day2_events.dat \
BBOX_PATH=data/annotations/outdoor_day2_motion_labels.npy \
DELTA_T=22858 \
bash quickstart.sh
```

## Prophesee 1 Megapixel Automotive Detection Dataset

Download from [prophesee.ai](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/).

> **Note:** The dataset is distributed in chunks. The four sequences used in this
> work are all from the **`trainfilelist14`** chunk.

Bounding boxes from the official dataset annotations were used as the spatial basis;
motion-state labels were assigned manually on top following the class scheme above.

The four sequences used in the paper, with their corresponding annotation files:

| Event file | Annotation |
|---|---|
| `moorea_2019-02-18_000_td_61500000_121500000_td.dat` | `data/annotations/moorea_2019-02-18_000_td_61500000_121500000_motion_labels.npy` |
| `moorea_2019-02-19_004_td_244500000_304500000_td.dat` | `data/annotations/moorea_2019-02-19_004_td_244500000_304500000_motion_labels.npy` |
| `moorea_2019-04-18_test_03_000_366500000_426500000_td.dat` | `data/annotations/moorea_2019-04-18_test_03_000_366500000_426500000_motion_labels.npy` |
| `moorea_2019-06-26_test_02_000_976500000_1036500000_td.dat` | `data/annotations/moorea_2019-06-26_test_02_000_976500000_1036500000_motion_labels.npy` |

Example run (first sequence):
```bash
DATASET=prophesee \
EVENT_PATH=data/moorea_2019-02-18_000_td_61500000_121500000_td.dat \
BBOX_PATH=data/annotations/moorea_2019-02-18_000_td_61500000_121500000_motion_labels.npy \
DELTA_T=16666 \
bash quickstart.sh
```
