# Event-based Motion Segmentation with Graph Transformer Neural Network

## EMS-GTNN 
In this paper, inspired by the success of the ED-KoGTL , we propose the Dynamic Object Mask-aware Event Labeling (DOMEL), which is an offline approach for annotating event data for motion segmentation applications. Every event in the recorded stream is assigned a label; foreground event or background event. The labeling process requires as input the corresponding gray-scale frame; which can be captured using a frame-based sensor, working simultaneously alongside the event camera. Hence, DOMEL allows labeling event streams recorded using event cameras that do not generate gray scale images such as DVXplorer. The frame-based sensor and the event camera need to visualize the same scene, with the same field of view. DOMEL approach includes four main stages, event-image synchronization, raw event-edge fitting, spatially shifted event-mask fitting, and event labeling, as illustrated in the following:

![MAINCOMPONENTSOFVISUAL-LOCALIZATION](https://github.com/Yusra-alkendi/EMS-GTNN/blob/main/DomelFramework.png)



![grab-landing-page](https://github.com/Yusra-alkendi/EventDenoising_GNNTransformer/blob/f1d9cdab93facdf39861fe72c409b1bb5aa25290/Dataset_Goodlight_750lux.gif)

![grab-landing-page](https://github.com/Yusra-alkendi/EventDenoising_GNNTransformer/blob/c2d36cf409546c44dc055122cb114d70ed4d5a02/Dataset_Lowlight_5lux.gif)

## Dynamic Object Mask-aware Event Labeling (DOMEL) Framework for Event-based Motion Segmentation (EMS)

DOMEL is a novel event labelling methodology developed to classify events, acquired when the camera is in motion, into two main classes: foreground or background events. The proposed DOMEL works irrespective of the sensor resolution, and hence any event camera may be used to record the event streams. A frame-based sensor is needed to capture intensity images corresponding to the recorded events, which will assist event labeling.
Two dynamic active pixel vision sensors; DAVIS346C and DVXplorer, are mounted side-by-side on a tripod, to capture a dynamic scene. DAVIS346C and DVXplorer have a spatial resolution of 346x260 and 640x480, a bandwidth of 12 and 165 MEvent/s, and a dynamic range of 120 and 90 dB, respectively. The cameras are moved along various trajectories; i.e. a sequence of translations and rotations, in environments with various scene geometries where objects of various types and sizes are randomly moving. 

Three measurements were recorded; (1) DAVIS346C event streams, (2) DVXplorer event streams, and (3) Gray scale images which capture intensity measurements of the dynamic scene. The gray scale images are obtained from the frame output of the DAVIS346C sensor and are denoted as active pixel sensor (APS) images hereafter. It is worth noting that in absence of the frame output of the event camera, it is possible to use a standard camera to capture the same scene alongside the event camera



## Event Denoising Known-object Ground-Truth Labeling (ED-KoGTL) dataset - Files

Row experimental data:

  **(1)** **"RawDVS_ExperimentalData":** raw sensor data is in ".mat" format. 

The labelled dataset for each experimental scenarios:

  **(2)** **"Dataset_Goodlight_750lux":** contains labeled event dataset, "Dataset_Goodlight_750lux.mat", of ∼750lux (Good light). 
After loading the file in MATLAB. You will find
  - "Dataset_Goodlight_750lux.x" and "Dataset_Goodlight_750lux.y" indicate the pixel coordinates at which the event occurred. 
  - "Dataset_Goodlight_750lux.t" indicates the event’s timestamp
  - "Dataset_Goodlight_750lux.label" indicate the event’s Label as 1 (real activity event) or 0 (noise).

  **(3)** **"Dataset_Lowlight_5lux":** contains labeled event dataset of ∼5lux (Low light). 
After loading the file in MATLAB. You will find
  - "Dataset_Lowlight_5lux.x" and "Dataset_Lowlight_5lux.y" indicate the pixel coordinates at which the event occurred. 
  - "Dataset_Lowlight_5lux.t" indicates the event’s timestamp
  - "Dataset_Lowlight_5lux.label" indicate the event’s Label as 1 (real activity event) or 0 (noise).


For additional information please see the paper and <https://youtu.be/x0FXZLEenJ8>.


# Additional Notes
Feel free to contact the repository owner if you need any help with using the Labelled dataset <yusra.alkendi@ku.ac.ae>. 

