"""
High-Fidelity RSNA-Compatible DICOM & CSV Synthetic Data Generator.
Creates authentic DICOM files with real MRI anatomy simulation and corresponding RSNA CSV files:
- train.csv
- train_series_descriptions.csv
- train_label_coordinates.csv
- train_images/<study_id>/<series_id>/<instance_number>.dcm
"""
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import cv2

from config.config import CONFIG

def simulate_mri_slice(
    series_type: str, 
    level_idx: int, 
    severity: str, 
    slice_idx: int, 
    total_slices: int = 15, 
    img_size: int = 512
) -> np.ndarray:
    """
    Generates a high-fidelity synthetic lumbar spine MRI slice (512x512) with authentic
    anatomical structures:
    - Vertebral bodies (L1 through S1) with physiological lordosis and trabecular marrow
    - Intervertebral discs with hydrated nucleus pulposus and annular margins
    - Continuous dural sac with high-signal CSF (T2) and cauda equina nerve fibers
    - Pathological stenosis narrowing (disc bulge + ligamentum flavum hypertrophy)
    - Posterior spinous processes, dorsal subcutaneous fat, and anterior soft tissues
    """
    img = np.zeros((img_size, img_size), dtype=np.float32)
    center_x = img_size // 2
    center_y = img_size // 2

    # Realistic MRI background thermal noise
    noise = np.random.normal(18, 4, (img_size, img_size)).astype(np.float32)
    img += noise

    if "Sagittal" in series_type:
        # SAGITTAL LUMBAR SPINE VIEW
        vert_height = 46
        disc_height = 18
        vert_width = 86
        start_y = 50

        # Lumbar lordosis curvature offsets for 6 segments (L1, L2, L3, L4, L5, S1)
        # Vertebrae curve anteriorly in mid-lumbar region (L3-L4)
        lordosis_offsets = [-15, -5, 8, 12, 0, -25]

        # Draw Vertebrae and Discs
        level_labels = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']
        canal_x_centers = []
        y_centers = []

        for i in range(6):
            y_v = start_y + i * (vert_height + disc_height)
            cx_offset = lordosis_offsets[i]
            vx1 = center_x - vert_width - 15 + cx_offset
            vx2 = center_x - 15 + cx_offset
            y_centers.append(y_v + vert_height // 2)

            # 1. Vertebral Body (Trabecular cancellous bone)
            # Bone marrow intensity: moderate on T2, bright on T1
            vert_base_color = 85.0 if "T2" in series_type else 115.0
            # Inner bone texture
            v_patch = np.random.normal(vert_base_color, 7, (vert_height, vert_width)).astype(np.float32)
            img[y_v:y_v + vert_height, vx1:vx2] = np.clip(v_patch, 40, 200)

            # Cortical bone rim (Dark/hypointense cortical shell)
            cv2.rectangle(img, (vx1, y_v), (vx2, y_v + vert_height), color=25.0, thickness=2)

            # 2. Intervertebral Disc Space (between vertebrae)
            if i < 5:
                y_d = y_v + vert_height
                dx1 = vx1 + 3
                dx2 = vx2 - 2

                # Disc annulus fibrosus (low-to-intermediate signal)
                disc_base = 65.0 if "T2" in series_type else 50.0
                cv2.rectangle(img, (dx1, y_d), (dx2, y_d + disc_height), color=disc_base, thickness=-1)

                # Nucleus pulposus in center (T2 hyperintense hydration signal)
                if "T2" in series_type:
                    # In degenerate/stenotic discs, nucleus loses hydration signal
                    nuc_intensity = 180.0 if (i != level_idx or severity == 'Normal/Mild') else (110.0 if severity == 'Moderate' else 75.0)
                    cv2.ellipse(img, ((dx1 + dx2) // 2, y_d + disc_height // 2), ((dx2 - dx1) // 3, disc_height // 3), 0, 0, 360, color=nuc_intensity, thickness=-1)

            # Canal posterior border tracking
            canal_x_centers.append(vx2 + 18)

        # 3. Continuous Dural Sac / Spinal Canal (CSF + Cauda Equina)
        slice_factor = max(0.3, 1.0 - abs(slice_idx - total_slices // 2) / (total_slices // 2))
        thecal_nominal_width = int(26 * slice_factor)
        csf_signal = 235.0 if "T2" in series_type else 45.0

        for i in range(len(y_centers) - 1):
            y_start = y_centers[i]
            y_end = y_centers[i + 1]
            x_start = canal_x_centers[i]
            x_end = canal_x_centers[i + 1]

            steps = 25
            for s in range(steps):
                cur_y = int(y_start + (y_end - y_start) * (s / steps))
                cur_x = int(x_start + (x_end - x_start) * (s / steps))
                
                # Check stenosis narrowing at target level disc space
                w = thecal_nominal_width
                target_disc_y = start_y + level_idx * (vert_height + disc_height) + vert_height + disc_height // 2
                dist_to_stenosis = abs(cur_y - target_disc_y)

                if dist_to_stenosis < (disc_height + 10):
                    factor = 1.0 - (dist_to_stenosis / (disc_height + 10))
                    if severity == 'Moderate':
                        w = int(w * (1.0 - 0.45 * factor))
                    elif severity == 'Severe':
                        w = int(w * (1.0 - 0.78 * factor))

                # Draw dural sac slice
                cv2.line(img, (cur_x - w // 2, cur_y), (cur_x + w // 2, cur_y), color=csf_signal, thickness=2)

                # Cauda equina nerve root filaments (thin dark dots/lines inside CSF on T2)
                if "T2" in series_type and w > 8:
                    for _ in range(2):
                        rx = cur_x + random.randint(-w // 3, w // 3)
                        img[cur_y, rx] = 50.0

        # If Stenosis present, draw posterior disc bulge & ligamentum flavum encroachment
        if severity in ['Moderate', 'Severe']:
            t_disc_y = start_y + level_idx * (vert_height + disc_height) + vert_height + disc_height // 2
            t_canal_x = canal_x_centers[level_idx]
            bulge_size = 14 if severity == 'Moderate' else 22
            # Anterior disc herniation / bulge (low signal)
            cv2.ellipse(img, (t_canal_x - thecal_nominal_width // 2, t_disc_y), (bulge_size // 2, disc_height // 2 + 3), 0, 0, 360, color=45.0, thickness=-1)
            # Posterior ligamentum flavum hypertrophy (low-intermediate signal)
            cv2.ellipse(img, (t_canal_x + thecal_nominal_width // 2, t_disc_y), (bulge_size // 2, disc_height // 2 + 5), 0, 0, 360, color=55.0, thickness=-1)

        # 4. Posterior Elements: Lamina, Spinous Processes, Paraspinous Muscle
        for i, cy in enumerate(y_centers):
            cx = canal_x_centers[i]
            # Posterior spinous process
            cv2.ellipse(img, (cx + 55, cy), (32, 14), -10, 0, 360, color=75.0, thickness=-1)
            cv2.ellipse(img, (cx + 55, cy), (32, 14), -10, 0, 360, color=30.0, thickness=2)

        # Dorsal subcutaneous fat layer (hyperintense on T1 & T2)
        cv2.rectangle(img, (center_x + 120, 30), (center_x + 160, 480), color=155.0, thickness=-1)
        # Anterior retroperitoneal fat and abdominal wall
        cv2.rectangle(img, (center_x - 220, 40), (center_x - 170, 470), color=135.0, thickness=-1)

        # Burn-in anatomical vertebral labels (L1 to S1) like PACS workstation
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        for i, (cy, lbl) in enumerate(zip(y_centers, level_labels)):
            vx = center_x - vert_width - 45 + lordosis_offsets[i]
            cv2.putText(img_u8, lbl, (vx, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 230, 1, cv2.LINE_AA)
        img = img_u8.astype(np.float32)

    else:
        # AXIAL LUMBAR MRI VIEW (512x512)
        # Anterior abdominal wall and visceral fat
        cv2.ellipse(img, (center_x, center_y - 95), (170, 95), 0, 0, 360, color=70.0, thickness=-1)
        cv2.ellipse(img, (center_x, center_y - 125), (195, 30), 0, 0, 360, color=130.0, thickness=-1)

        # Intervertebral Disc / Vertebral Endplate (kidney-bean / horseshoe shaped)
        cv2.ellipse(img, (center_x, center_y - 35), (115, 75), 0, 0, 360, color=105.0, thickness=-1)
        # Posterior disc concavity
        cv2.ellipse(img, (center_x, center_y + 10), (60, 35), 0, 0, 360, color=35.0, thickness=-1)

        # Spinal Canal (Thecal sac lumen)
        canal_center = (center_x, center_y + 40)
        canal_rx = 45
        canal_ry = 38

        # Apply stenosis constriction
        if severity == 'Moderate':
            canal_rx = int(canal_rx * 0.55)
            canal_ry = int(canal_ry * 0.50)
        elif severity == 'Severe':
            canal_rx = int(canal_rx * 0.28)
            canal_ry = int(canal_ry * 0.24)

        # Dural Sac CSF lumen (bright on T2)
        cv2.ellipse(img, canal_center, (canal_rx, canal_ry), 0, 0, 360, color=225.0, thickness=-1)

        # Cauda equina nerve root dots inside CSF
        if severity != 'Severe':
            for _ in range(8):
                rx = canal_center[0] + random.randint(-max(1, canal_rx - 8), max(1, canal_rx - 8))
                ry = canal_center[1] + random.randint(-max(1, canal_ry - 8), max(1, canal_ry - 8))
                cv2.circle(img, (rx, ry), 2, color=40.0, thickness=-1)

        # Facet joints and posterior neural arch
        cv2.line(img, (center_x - 70, center_y + 45), (center_x - 25, center_y + 95), color=40.0, thickness=12)
        cv2.line(img, (center_x + 70, center_y + 45), (center_x + 25, center_y + 95), color=40.0, thickness=12)
        # Spinous process
        cv2.ellipse(img, (center_x, center_y + 105), (30, 22), 0, 0, 360, color=65.0, thickness=-1)

    # Apply realistic MRI point spread function & noise
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    # Clip & scale to standard uint16 medical DICOM range
    img = np.clip(img * 16.0, 0, 4095).astype(np.uint16)
    return img

def create_dicom_file(
    file_path: Path, 
    pixel_array: np.ndarray, 
    patient_id: str, 
    study_uid: str, 
    series_uid: str, 
    instance_num: int, 
    series_description: str
):
    """Creates a fully valid DICOM Part 10 file with standard medical tags."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = str(patient_id)
    ds.PatientName = f"Patient^{patient_id}"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "MR"
    ds.SeriesDescription = series_description
    ds.InstanceNumber = instance_num
    ds.Rows, ds.Columns = pixel_array.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.8, 0.8]
    ds.SliceThickness = 3.0
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_num * 3.0)]
    ds.WindowCenter = 1000
    ds.WindowWidth = 2000
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.PixelData = pixel_array.tobytes()

    file_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(file_path), write_like_original=False)

def generate_mock_rsna_dataset(num_patients: int = 40, output_dir: Path = None):
    """
    Generates a full synthetic RSNA-format Lumbar Spine dataset with:
    - 40 patients (with train/test group splitting)
    - Realistic class imbalance: Normal/Mild (65%), Moderate (25%), Severe (10%)
    - 3 series per patient: Sagittal T2, Sagittal T1, Axial T2
    - Matching train.csv, train_series_descriptions.csv, train_label_coordinates.csv
    """
    if output_dir is None:
        output_dir = CONFIG.data_dir

    images_dir = output_dir / "train_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
    level_names = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    conditions = [
        'spinal_canal_stenosis',
        'left_neural_foraminal_stenosis',
        'right_neural_foraminal_stenosis',
        'left_subarticular_stenosis',
        'right_subarticular_stenosis'
    ]
    condition_names = [
        'Spinal Canal Stenosis',
        'Left Neural Foraminal Stenosis',
        'Right Neural Foraminal Stenosis',
        'Left Subarticular Stenosis',
        'Right Subarticular Stenosis'
    ]

    train_rows = []
    series_rows = []
    coords_rows = []

    print(f"Generating realistic RSNA Lumbar Spine dataset for {num_patients} patients...")

    # Class distribution weights: 65% Normal/Mild, 25% Moderate, 10% Severe
    severity_options = ['Normal/Mild', 'Moderate', 'Severe']
    severity_weights = [0.65, 0.25, 0.10]

    for p_idx in range(1, num_patients + 1):
        study_id = 100000000 + p_idx
        study_uid = generate_uid()

        # Generate severity labels for this patient across all conditions & levels
        patient_row = {'study_id': study_id}
        patient_severities = {}

        for cond in conditions:
            for lvl in levels:
                col_name = f"{cond}_{lvl}"
                sev = random.choices(severity_options, weights=severity_weights)[0]
                patient_row[col_name] = sev
                patient_severities[col_name] = sev

        train_rows.append(patient_row)

        # 3 MRI series per patient
        series_info = [
            (200000000 + p_idx * 10 + 1, 'Sagittal T2/STIR', 12),
            (200000000 + p_idx * 10 + 2, 'Sagittal T1', 12),
            (200000000 + p_idx * 10 + 3, 'Axial T2', 15)
        ]

        for s_id, s_desc, num_slices in series_info:
            series_uid = generate_uid()
            series_rows.append({
                'study_id': study_id,
                'series_id': s_id,
                'series_description': s_desc
            })

            # Create DICOM slices
            for inst in range(1, num_slices + 1):
                # Determine severity at this slice
                dominant_level = min(4, max(0, int((inst - 1) / (num_slices / 5))))
                lvl_key = levels[dominant_level]
                sev = patient_severities.get(f"spinal_canal_stenosis_{lvl_key}", "Normal/Mild")

                slice_arr = simulate_mri_slice(
                    series_type=s_desc,
                    level_idx=dominant_level,
                    severity=sev,
                    slice_idx=inst,
                    total_slices=num_slices,
                    img_size=512
                )

                dcm_path = images_dir / str(study_id) / str(s_id) / f"{inst}.dcm"
                create_dicom_file(
                    file_path=dcm_path,
                    pixel_array=slice_arr,
                    patient_id=str(study_id),
                    study_uid=study_uid,
                    series_uid=series_uid,
                    instance_num=inst,
                    series_description=s_desc
                )

                # Add coordinate annotation for key canal slices
                if "Sagittal T2" in s_desc and inst in [5, 6, 7]:
                    for l_idx, (l_code, l_display) in enumerate(zip(levels, level_names)):
                        y_pos = int(50 + l_idx * 64 + 32)
                        x_pos = int(256)
                        coords_rows.append({
                            'study_id': study_id,
                            'series_id': s_id,
                            'instance_number': inst,
                            'condition': 'Spinal Canal Stenosis',
                            'level': l_display,
                            'x': x_pos,
                            'y': y_pos
                        })

    # Save CSVs
    df_train = pd.DataFrame(train_rows)
    df_series = pd.DataFrame(series_rows)
    df_coords = pd.DataFrame(coords_rows)

    df_train.to_csv(output_dir / "train.csv", index=False)
    df_series.to_csv(output_dir / "train_series_descriptions.csv", index=False)
    df_coords.to_csv(output_dir / "train_label_coordinates.csv", index=False)

    print(f"Generated {len(df_train)} patient records.")
    print(f"Generated {len(df_series)} MRI series.")
    print(f"Generated {len(df_coords)} coordinate annotations.")
    print(f"Saved dataset files to {output_dir}")

    # Display class distribution
    canal_l4_l5 = df_train['spinal_canal_stenosis_l4_l5'].value_counts(normalize=True) * 100
    print("\nSample Class Distribution (Spinal Canal Stenosis L4/L5):")
    for cls_name, pct in canal_l4_l5.items():
        print(f"  {cls_name}: {pct:.1f}%")

    return df_train, df_series, df_coords

if __name__ == "__main__":
    generate_mock_rsna_dataset(num_patients=24)
