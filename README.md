# Event-based Motion Segmentation with Graph Transformer Neural Network

## EMS-GTNN 
In this paper, inspired by the success of the ED-KoGTL , we propose the Dynamic Object Mask-aware Event Labeling (DOMEL), which is an offline approach for annotating event data for motion segmentation applications. Every event in the recorded stream is assigned a label; foreground event or background event. The labeling process requires as input the corresponding gray-scale frame; which can be captured using a frame-based sensor, working simultaneously alongside the event camera. Hence, DOMEL allows labeling event streams recorded using event cameras that do not generate gray scale images such as DVXplorer. The frame-based sensor and the event camera need to visualize the same scene, with the same field of view. DOMEL approach includes four main stages, event-image synchronization, raw event-edge fitting, spatially shifted event-mask fitting, and event labeling, as illustrated in the following:

![MAINCOMPONENTSOFVISUAL-LOCALIZATION](https://github.com/Yusra-alkendi/EMS-GTNN/blob/main/DomelFramework.png)



![grab-landing-page](https://github.com/Yusra-alkendi/EMS-GTNN/blob/main/EMS_DOMEL_SEQ01.gif)

![grab-landing-page](https://github.com/Yusra-alkendi/EMS-GTNN/blob/main/EMS_DOMEL_SEQ02.gif)

![grab-landing-page](https://github.com/Yusra-alkendi/EMS-GTNN/blob/main/EMS_DOMEL_SEQ03.gif)



## Dynamic Object Mask-aware Event Labeling (DOMEL) Framework for Event-based Motion Segmentation (EMS)

DOMEL is a novel event labelling methodology developed to classify events, acquired when the camera is in motion, into two main classes: foreground or background events. The proposed DOMEL works irrespective of the sensor resolution, and hence any event camera may be used to record the event streams. A frame-based sensor is needed to capture intensity images corresponding to the recorded events, which will assist event labeling.
Two dynamic active pixel vision sensors; DAVIS346C and DVXplorer, are mounted side-by-side on a tripod, to capture a dynamic scene. DAVIS346C and DVXplorer have a spatial resolution of 346x260 and 640x480, a bandwidth of 12 and 165 MEvent/s, and a dynamic range of 120 and 90 dB, respectively. The cameras are moved along various trajectories; i.e. a sequence of translations and rotations, in environments with various scene geometries where objects of various types and sizes are randomly moving. 

Three measurements were recorded; (1) DAVIS346C event streams, (2) DVXplorer event streams, and (3) Gray scale images which capture intensity measurements of the dynamic scene. The gray scale images are obtained from the frame output of the DAVIS346C sensor and are denoted as active pixel sensor (APS) images hereafter. It is worth noting that in absence of the frame output of the event camera, it is possible to use a standard camera to capture the same scene alongside the event camera



## Event-based Motion Segmentation (EMS) dataset using Dynamic Object Mask-aware Event Labeling (DOMEL) approach - EMS-DOMEL Files

The labelled dataset for each  scenarios:

  **(1)** **"DOMEL_Seq01":** contains labeled event dataset for motion segmentation, "DOMEL_Seq01.mat". 
After loading the file in MATLAB. You will find
  - "DOMEL_Seq01.x" and "DOMEL_Seq01.y" indicate the pixel coordinates at which the event occurred. 
  - "DOMEL_Seq01.t" indicates the event’s timestamp
  - "DOMEL_Seq01.p" indicates the event’s polarity
  - "DOMEL_Seq01x.label" indicate the event’s Label as 1 (Foreground event) or 0 (Background event).


  **(2)** **"DOMEL_Seq02":** contains labeled event dataset for motion segmentation, "DOMEL_Seq02.mat". 
After loading the file in MATLAB. You will find
  - "DOMEL_Seq02.x" and "DOMEL_Seq02.y" indicate the pixel coordinates at which the event occurred. 
  - "DOMEL_Seq02.t" indicates the event’s timestamp
  - "DOMEL_Seq02.p" indicates the event’s polarity
  - "DOMEL_Seq02.label" indicate the event’s Label as 1 (Foreground event) or 0 (Background event).

  **(3)** **"DOMEL_Seq03":** contains labeled event dataset for motion segmentation, "DOMEL_Seq03.mat". 
After loading the file in MATLAB. You will find
  - "DOMEL_Seq03.x" and "DOMEL_Seq03.y" indicate the pixel coordinates at which the event occurred. 
  - "DOMEL_Seq03.t" indicates the event’s timestamp
  - "DOMEL_Seq03.p" indicates the event’s polarity
  - "DOMEL_Seq03.label" indicate the event’s Label as 1 (Foreground event) or 0 (Background event).




For additional information please see the paper and <You_tube_link_video>.


# Additional Notes
Feel free to contact the repository owner if you need any help with using the Labelled dataset <yusra.alkendi@ku.ac.ae>. 

