# sub-region-mri-segmentation

MRI sub-region segmentation using SAM and Ants segmentation methods <br/>
We were provided with an atlas of the marmoset brain dataset, and our goal was to improve the current atlas. <br/>
dataset: "sym_avg_mri_2023_50mu.nii.gz" and "sym_pop_atlas_label_map_2023.nii.gz"
provided by the Brain/MIND project. https://dataportal.brainminds.jp/<br/>

We selected three ROIs to work on, including 233-Caudate, 234-Putamen, 633-Anterior Commissure, different methods to segment and refine the current atlas annotation, including the Segment Anything Model (SAM), all the segmentation methods provided by ANTs<br/>

The following shows a sample slice of the dataset and its corresponding labels. The 3D representation of the refined ROIs is also provided in the following figure (obtained from SAM)<br/>

![dataset-labels](https://github.com/shahrokh1106/sub-region-mri-segmentation-main/assets/44213732/3a12e370-df32-44e8-aa35-45e7924e7178)

![results3D](https://github.com/shahrokh1106/sub-region-mri-segmentation-main/assets/44213732/16547232-7413-4a4d-883f-1500dccc1a74)
