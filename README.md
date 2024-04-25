# sub-region-mri-segmentation

MRI sub-region segmentation using SAM and Ants segmentation methods <br/>
We were provided with an atlas of the marmoset brain dataset, and our goal was to improve the current atlas. <br/>
dataset: "sym_avg_mri_2023_50mu.nii.gz" and "sym_pop_atlas_label_map_2023.nii.gz"
provided by the Brain/MIND project. https://dataportal.brainminds.jp/<br/>

We selected three ROIs to work on, including 233-Caudate, 234-Putamen, 633-Anterior Commissure, different methods to segment and refine the current atlas annotation, including the Segment Anything Model (SAM), all the segmentation methods provided by ANTs<br/>

The following shows a sample slice of the dataset and its corresponding labels. The 3D representation of the refined ROIs is also provided in the following figure (obtained from SAM)<br/>

![dataset-labels](https://github.com/shahrokh1106/sub-region-mri-segmentation-main/assets/44213732/3a12e370-df32-44e8-aa35-45e7924e7178)

![results3D](https://github.com/shahrokh1106/sub-region-mri-segmentation-main/assets/44213732/16547232-7413-4a4d-883f-1500dccc1a74)


To get inference from the SAM model on the marmoset dataset, use SAMmain.py, where
*nifti_image_path: is the path to the marmoset average mri dataset 
*nifti_label_path: is the current atlas label for the mri dataset
*offset: there is an offset of 10000 between the lebel values of the left and right brian hemispheres
*ID: is thw id or lable value of the ROI
*device: If GPU is available
*sam_model_type: is the SAM model type
*sam_checkpoint: is the address of the corresponding SAM pretrianed model
*mode_axis: is the selected axis for slicing ("axial", "coronal", "sagittal")
*write_results: if true, after each run the results will be saved in the current directoy as a nifti image data
