# Color_detection_histograms

## This repository includes:

  • a histogram calculator
  • a color detection classifier based on histogram numpy arrays (neural networks + svm)
  • a color detection classifier based on histogram images (CNN + svm)
  
  
## Histogram Calculator
 
 your dataset directory should be in this format:
 
 	Dataset 
 	      ├── train
 	      │   │
	      │   ├── blue 
	      │   │     ├── image1.jpg
	      │   │     └── image2.jpg
	      │   │
	      │   ├── red 
	      │   │     ├── image1.jpg
	      │   │     └── image2.jpg 
	      │   │
	      │   └── yellow
	      │         ├── image1.jpg
	      │         └── image2.jpg 
	      │
	      │
	      └── test
	          │
	          ├── blue 
	          │     ├── image1.jpg
	          │     └── image2.jpg
	          │
	          ├── red 
	          │     ├── image1.jpg
	          │     └── image2.jpg 
	          │
	          └── yellow
	                ├── image1.jpg
	                └── image2.jpg 
 
 
  just insert your dataset and output path in create_hist_data.py then run it. it will generate calculated histograms for your dataset.
  
## Color detection

There is 2 way to classifiy your histogram dataset:
	-as image
	-as numerical arrays
	
for each of them there are data_loader.py and classifier.py
you can easily insert the path of histograms numpys in dataloader, then run the classifier.
  
  
