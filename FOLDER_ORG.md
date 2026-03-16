Organization and Contents of SP-NTL folder.



**data** - contains the 2000 NTL images used in the SR pipeline

&#x20; test\_data - contains the collection of images used as test data



**Helper\_Scripts**- helper code for various applications

&#x20; analyze\_results - program that takes the raw data from the SR\_pipeline and outputs usefull csv and image collections

&#x20; select\_testing\_data - program to partition the data into train/validation data and testing data



**SR\_pipeline** - code for the SR pipeline

&#x20; dataset - loads the train and validation data, performs preprossing (downsampling) and validation and training partitioning

&#x20; SR\_NTL\_Semester\_Project - training code for the SR pipeline

&#x20; test - run the trained learned SR model on the test data as well as through bicubic interpolation



**runs\_sr** - contains the output of the model, saved check points and the analysis output

&#x20; analysis - output of the analyze\_results script

&#x20; test\_raw\_output - output of the test code: bicubic and learned metrics, saved bicubic and learned visuals

&#x20; training\_output - output of the training pipeline: validation metrics, learning curves visuals, validation visuals, model check point and val/train split

&#x20;    

**data\_collection** - world cities database, selected samples, helper programs, TIF tile footprints, GEE script



**initial\_debugging\_data** - Contains the intial NTL images used to debug the SR pipeline.



