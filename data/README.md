## Prepare datasets ##
The proposed method ISCL was verified on two datasets (i.e., CT and EM)
1. Normal-dose CT and Low-dose CT from [this site](https://www.aapm.org/grandchallenge/LowDoseCT/)
- A library of CT image and projection data is publicly available [here](https://www.cancerimagingarchive.net/)

You can download the same dataset using follow subject IDs:
* Abdominal
    * Subject IDs: 
        * Training
            * L004, L006, L014, L019, L033, L049, L056, L057, L058, L064, L071, L072, L075, L077, L081, L107, L110, L114, L116, L123
        * Testing
            * L125, L131, L134, L145, L148, L150, L160, L170, L175, L178

* Chest
    * Subject IDs: 
        * Training
            * C002, C004, C012, C016, C027, C030, C050, C052, C067, C077, C081, C095, C099, C107, C111, C120, C121, C124, C128, C135
        * Testing
            * C158, C160, C162, C166, C170, C179, C190, C193, C202, C203
        
2. EM dataset of charge noise and film noise first used in [Quan et al.](https://openaccess.thecvf.com/content_ICCVW_2019/papers/LCI/Quan_Removing_Imaging_Artifacts_in_Electron_Microscopy_using_an_Asymmetrically_Cyclic_ICCVW_2019_paper.pdf)
- All dataset used in this paper is available [here](https://drive.google.com/file/d/1yberw2R3HeKLp79tRtWoMZjLQMSmTCIS/view?usp=sharing)

After download the EM dataset, you can check below lists as we described in the paper.
* TEM_ZB (tem) (clean)
    * The TEM_ZB images were captured at a resolution of 4.0 X 4.0 X ~ 40nm^3vx^-1 using a modified JEOL 1200CX system.
* TEM_DR5 (dr5) (clean)
    * The TEM_DR5 was acquired from mouse visual cortex tissue sections collected onto pioloform support film.
* SEM_ZB (sem) (noisy)
    * The SEM_ZB images were captured at a resolution of 4.0 X 4.0 X ~ 60nm^3vx^-1 using a FEI Magellan XHR400L system.
* TEM_PPC (ppc) (noisy)
    * The TEM_PPC was acquired from mouse posterior parietal cortex tissue sections collected onto LUXfilm® support film (Luxel Corporation).
* TEM_ZB + Charge noise (simulated_image/tem_charge_noise)
    * This synthetic images were created by pixel-wise multiplication of film noise (i.e. captured images of lacking tissue sections) and TEM_ZB.
* TEM_DR5 + Film noise (simulated_image/dr5_film_noise)
    * This synthetic images were created by pixel-wise addition of charge noise (i.e. captured images of lacking tissue sections) and TEM_ZB.

To verify the proposed model in a realistic setup, we found the best hyper-parameters from the cross-validation of synthetic simulation.
