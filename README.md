# CAFFE for Hourglass

## Reference

>   Stacked Hourglass Networks for Human Pose Estimation : https://arxiv.org/abs/1603.06937  

>   Human Pose estimation with pytorch : https://github.com/bearpaw/pytorch-pose  
    
## Package needs  

```Shell
  Caffe (Train, Test, Inference)
  pytorch (Only for Data preparation)
```  
 Actually, I'm trying to implement for Data preparation only with requirements in Caffe, if finished, I will update this part.    
 
 
## Data preparation (MPII Dataset)

> for your own data, there's some needs to data preparation  


   It needs a text file for label like(format):  
  
  ```Shell
    name(for <name>.jpg file)  xmax  xmin  ymax  ymin  label
  ```  
  
  
   If you prepared a label text file like above, you can use a python script file data/yolo/conv2xml.py.
     
   It converts the label text to xml file (to PASCAL VOC format).
     
   Please sure that the images are in folder 'Images/', and you have to change some codes in main func (maybe file paths)    
        
  ```Shell
    python conv2xml.py
  ```
    
  And If you already have .xml files but doesn't fit to PASCAL VOC format, then try with data/yolo/conv2voc.py.  
  
  you have to prepare :  
  ```shell
    Main/Images : images
    Main/xml : empty (converted xml path)
    Main/xml2 : your own xml files
  ```  
  (Please remember that you have to change the code where it reads .xml files for node names. Change those parts for your own .xml)
    
  Then,  
  ```shell
    python conv2voc.py -f <Main Dir>
  ```
  

### + Inference for movie (if any video files)  

This file is inherited from show_det.py in examples/yolo.  

It needs a pretrained .caffemodel & deploy.prototxt file.    

And please remind that you have to set your own file's path in examples/yolo/yolo_video.py.  

```shell
  python3 yolo_video.py
```  





