\n\n#---------------------------------
# New invocation of recon-all Sat Jan 24 13:42:35 EST 2026 
\n mri_convert /Users/chakri/ishu/adni/data/002_S_0729/2006-08-02_07_02_00.0/2006-08-02_07_02_00.0.nii.gz /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/orig/001.mgz \n
#--------------------------------------------
#@# MotionCor Sat Jan 24 13:42:42 EST 2026
\n cp /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/orig/001.mgz /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/rawavg.mgz \n
\n mri_convert /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/rawavg.mgz /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/orig.mgz --conform \n
\n mri_add_xform_to_header -c /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/transforms/talairach.xfm /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/orig.mgz /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/orig.mgz \n
#--------------------------------------------
#@# Talairach Sat Jan 24 13:42:50 EST 2026
\n mri_nu_correct.mni --no-rescale --i orig.mgz --o orig_nu.mgz --n 1 --proto-iters 1000 --distance 50 \n
\n talairach_avi --i orig_nu.mgz --xfm transforms/talairach.auto.xfm \n
talairach_avi log file is transforms/talairach_avi.log...
\n cp transforms/talairach.auto.xfm transforms/talairach.xfm \n
#--------------------------------------------
#@# Talairach Failure Detection Sat Jan 24 13:44:36 EST 2026
\n talairach_afd -T 0.005 -xfm transforms/talairach.xfm \n
\n awk -f /Applications/freesurfer/bin/extract_talairach_avi_QA.awk /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/transforms/talairach_avi.log \n
\n tal_QC_AZS /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/transforms/talairach_avi.log \n
#--------------------------------------------
#@# Nu Intensity Correction Sat Jan 24 13:44:37 EST 2026
\n mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --n 2 \n
\n mri_add_xform_to_header -c /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/mri/transforms/talairach.xfm nu.mgz nu.mgz \n
#--------------------------------------------
#@# Intensity Normalization Sat Jan 24 13:46:34 EST 2026
\n mri_normalize -g 1 -mprage nu.mgz T1.mgz \n
#--------------------------------------------
#@# Skull Stripping Sat Jan 24 13:48:53 EST 2026
\n mri_em_register -rusage /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/touch/rusage.mri_em_register.skull.dat -skull nu.mgz /Applications/freesurfer/average/RB_all_withskull_2016-05-10.vc700.gca transforms/talairach_with_skull.lta \n
\n mri_watershed -rusage /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/touch/rusage.mri_watershed.dat -T1 -brain_atlas /Applications/freesurfer/average/RB_all_withskull_2016-05-10.vc700.gca transforms/talairach_with_skull.lta T1.mgz brainmask.auto.mgz \n
\n cp brainmask.auto.mgz brainmask.mgz \n
#-------------------------------------
#@# EM Registration Sat Jan 24 14:03:02 EST 2026
\n mri_em_register -rusage /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/touch/rusage.mri_em_register.dat -uns 3 -mask brainmask.mgz nu.mgz /Applications/freesurfer/average/RB_all_2016-05-10.vc700.gca transforms/talairach.lta \n
#--------------------------------------
#@# CA Normalize Sat Jan 24 14:15:18 EST 2026
\n mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /Applications/freesurfer/average/RB_all_2016-05-10.vc700.gca transforms/talairach.lta norm.mgz \n
#--------------------------------------
#@# CA Reg Sat Jan 24 14:56:24 EST 2026
\n mri_ca_register -rusage /Users/chakri/ishu/adni/freesurfer_subjects/002_S_0729_bl/touch/rusage.mri_ca_register.dat -nobigventricles -T transforms/talairach.lta -align-after -mask brainmask.mgz norm.mgz /Applications/freesurfer/average/RB_all_2016-05-10.vc700.gca transforms/talairach.m3z \n
